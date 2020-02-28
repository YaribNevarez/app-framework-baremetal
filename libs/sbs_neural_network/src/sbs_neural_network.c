/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: Yarib Nevarez
 */


#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "stddef.h"
#include "stdarg.h"

#include "sbs_neural_network.h"
#include "mt19937int.h"

#ifdef USE_XILINX
#include "ff.h"
#endif

#define ASSERT(expr)  assert(expr)

/*****************************************************************************/
#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */


typedef float     Weight;
typedef uint16_t  SpikeID;

typedef struct
{
  void *   data;
  size_t   data_type_size;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
} Multivector;

typedef struct
{
  SbsLayer      vtbl;

  Multivector * state_matrix;
  Multivector * weight_matrix;
  Multivector * spike_matrix;
  NeuronState * update_buffer;
  uint16_t      kernel_size;
  uint16_t      kernel_stride;
  uint16_t      neurons_previous_Layer;
  WeightShift   weight_shift;
  float         epsilon;
} SbsBaseLayer;

typedef struct
{
  SbsNetwork        vtbl;
  uint8_t           size;
  SbsBaseLayer **   layer_array;
  uint8_t           input_label;
  uint8_t           inferred_output;
} SbsBaseNetwork;


#pragma pack(pop)   /* restore original alignment from stack */

/*****************************************************************************/
/************************ Memory manager *************************************/
#define        MEMORY_SIZE    4763116

static size_t  Memory_blockIndex = 0;

static void * Memory_requestBlock(size_t size)
{
  static uint8_t Memory_block[MEMORY_SIZE];
  void * ptr = NULL;

  if (Memory_blockIndex + size <= sizeof(Memory_block))
  {
    ptr = (void *) &Memory_block[Memory_blockIndex];
    Memory_blockIndex += size;
  }

  return ptr;
}

static size_t Memory_getBlockSize(void)
{
  return Memory_blockIndex;
}

/*****************************************************************************/
/*****************************************************************************/

static Multivector * Multivector_new(uint8_t data_type_size, uint8_t dimensionality, ...)
{
  Multivector * multivector = NULL;

  ASSERT(0 <= dimensionality);

  if (0 <= dimensionality)
  {
    size_t memory_size = sizeof(Multivector) + (dimensionality - 1) * sizeof(uint16_t);
    multivector = malloc(memory_size);

    ASSERT(multivector != NULL);

    if (multivector != NULL)
    {
      int arg;
      size_t data_size;
      va_list argument_list;

      memset(multivector, 0x00, memory_size);

      va_start(argument_list, dimensionality);

      for (data_size = 1, arg = 0; arg < dimensionality; arg ++)
        data_size *= (multivector->dimension_size[arg] = (uint16_t) va_arg(argument_list, int));

      va_end(argument_list);

      multivector->data = Memory_requestBlock(data_size * data_type_size);

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size * data_type_size);

      multivector->dimensionality = dimensionality;
      multivector->data_type_size = data_type_size;
    }
  }

  return multivector;
}

static void Multivector_delete(Multivector ** multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(*multivector != NULL);

  if ((multivector != NULL) && (*multivector != NULL))
  {
    free(*multivector);
    *multivector = NULL;
  }
}

void * Multivector_2DAccess(Multivector * multivector, uint16_t row, uint16_t column)
{
  void * data = NULL;
  ASSERT (multivector != NULL);
  ASSERT (multivector->data != NULL);
  ASSERT (2 <= multivector->dimensionality);
  ASSERT (row <= multivector->dimension_size[0]);
  ASSERT (column <= multivector->dimension_size[1]);

  if ((multivector != NULL)
      && (multivector->data != NULL)
      && (2 <= multivector->dimensionality)
      && (row <= multivector->dimension_size[0])
      && (column <= multivector->dimension_size[1]))
  {
    uint16_t dimensionality = multivector->dimensionality;
    size_t data_size = multivector->data_type_size;

    while (dimensionality-- > 2)
    {
      data_size *= multivector->dimension_size[dimensionality];
    }

    data = multivector->data
        + (row * multivector->dimension_size[1] + column) * data_size;
  }

  return data;
}

static void Multivector_saveToCSV(Multivector * multivector, char * file_name)
{
  ASSERT(multivector != NULL);
  ASSERT(file_name != NULL);
  if ((multivector != NULL) && (file_name != NULL))
  {
    FILE * file = fopen (file_name, "w");
    ASSERT(file != NULL);

    if (file != NULL)
    {
      int row;
      int column;
      char * cell_ending;

      for (row = 0; row < multivector->dimension_size[0]; row++)
      {
        for (column = 0; column < multivector->dimension_size[1]; column++)
        {
          cell_ending =
              (column == multivector->dimension_size[1] - 1) ? "\n" : ",";

          fprintf (file, "%d%s",
              *(SpikeID *) Multivector_2DAccess (multivector, row, column),
              cell_ending);
        }
      }
      fclose(file);
    }
  }
}
/*****************************************************************************/
/*****************************************************************************/

static SbsLayer * SbsBaseLayer_new(uint16_t rows,
                                   uint16_t columns,
                                   uint16_t neurons,
                                   uint16_t kernel_size,
                                   uint16_t kernel_stride,
                                   WeightShift weight_shift,
                                   uint16_t    neurons_previous_Layer)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);

  if (layer != NULL)
  {
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    /* Instantiate state_matrix */
    state_matrix = Multivector_new(sizeof(NeuronState), 3, rows, columns, neurons);

    ASSERT(state_matrix != NULL);
    ASSERT(state_matrix->dimensionality == 3);
    ASSERT(state_matrix->data != NULL);
    ASSERT(state_matrix->dimension_size[0] == rows);
    ASSERT(state_matrix->dimension_size[1] == columns);
    ASSERT(state_matrix->dimension_size[2] == neurons);

    layer->state_matrix = state_matrix;

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new(sizeof(SpikeID), 2, rows, columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    layer->spike_matrix = spike_matrix;

    /* Allocate update buffer */

    layer->update_buffer = malloc(neurons * sizeof(NeuronState));

    ASSERT(layer->update_buffer != NULL);

    if (layer->update_buffer != NULL)
    	memset(layer->update_buffer, 0x00, neurons * sizeof(NeuronState));

    /* Assign parameters */
    layer->kernel_size   = kernel_size;
    layer->kernel_stride = kernel_stride;
    layer->weight_shift  = weight_shift;
    layer->neurons_previous_Layer = neurons_previous_Layer;
  }

  return (SbsLayer *) layer;
}

static void SbsBaseLayer_delete(SbsLayer ** layer_ptr)
{
  ASSERT(layer_ptr!= NULL);
  ASSERT(*layer_ptr!= NULL);
  if ((layer_ptr!= NULL) && (*layer_ptr!= NULL))
  {
    SbsBaseLayer ** layer = (SbsBaseLayer **)layer_ptr;
    Multivector_delete(&((*layer)->state_matrix));
    Multivector_delete(&((*layer)->spike_matrix));
    if ((*layer)->weight_matrix != NULL) Multivector_delete(&((*layer)->weight_matrix));
    free((*layer)->update_buffer);
    free(*layer);
    *layer = NULL;
  }
}


static void SbsBaseLayer_initializeIP(NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
      float    initial_value_h = 1.0f / size;
      uint16_t neuron;
      for (neuron = 0; neuron < size; neuron ++)
        state_vector[neuron] = initial_value_h;
  }
}

#define FP_OR_MASK  (0x30000000)
#define FP_AND_MASK (0x0ffff000) | FP_OR_MASK


#define OR_MASK(reg)  {*((uint32_t*)&reg) |= FP_OR_MASK;}
#define AND_MASK(reg) {*((uint32_t*)&reg) &= FP_AND_MASK;}

#define FORCE(reg) {AND_MASK(reg); if (*((uint32_t*)&reg) != 0) while ((*((uint32_t*)&reg) & FP_OR_MASK) != FP_OR_MASK);}


static void SbsBaseLayer_updateIP(SbsBaseLayer * layer, NeuronState * state_vector, Weight * weight_vector, uint16_t size, float epsilon)
{
  ASSERT(state_vector != NULL);
  ASSERT(weight_vector != NULL);
  ASSERT(layer->update_buffer != NULL);
  ASSERT(0 < size);

  for (int i = 0; i < size; i ++)
  {
    FORCE(state_vector[i]);
    FORCE(weight_vector[i]);
  }

  FORCE(epsilon);

  if ((state_vector != NULL) && (weight_vector != NULL)
      && (layer->update_buffer != NULL) && (0 < size))
  {
    NeuronState * temp_data     = layer->update_buffer;

    NeuronState sum             = 0.0f;
    NeuronState reverse_epsilon = 1.0f / (1.0f + epsilon);
    NeuronState epsion_over_sum = 0.0f;
    uint16_t    neuron;

    FORCE(reverse_epsilon);

#if defined (__x86_64__) || defined(__amd64__)
    for (neuron = 0; neuron < size; neuron ++)
    {
      temp_data[neuron] = state_vector[neuron] * weight_vector[neuron];
      FORCE(temp_data[neuron]);

      sum += temp_data[neuron];
      FORCE(sum);
    }

    if (sum < 1e-20) // TODO: DEFINE constant
      return;

    epsion_over_sum = epsilon / sum;

    //FORCE(epsion_over_sum);

    for (neuron = 0; neuron < size; neuron ++)
    {
      state_vector[neuron] = reverse_epsilon * (state_vector[neuron] + temp_data[neuron] * epsion_over_sum);
      FORCE(state_vector[neuron]);
    }

#elif defined(__arm__)
    /* Support for unaligned accesses in ARM architecture */
    NeuronState h;
    NeuronState p;
    NeuronState h_p;
    NeuronState h_new;

    for (neuron = 0; neuron < size; neuron ++)
    {
      h = state_vector[neuron];
      p = weight_vector[neuron];
      h_p = h * p;

      temp_data[neuron] = h_p;
      sum += h_p;
    }

    if (sum < 1e-20) // TODO: DEFINE constant
      return;

    epsion_over_sum = epsilon / sum;

    for (neuron = 0; neuron < size; neuron ++)
    {
      h_p = temp_data[neuron];
      h = state_vector[neuron];

      h_new = reverse_epsilon * (h + h_p * epsion_over_sum);
      state_vector[neuron] = h_new;
    }
#else
#error "Unsupported processor architecture"
#endif
  }
}

static SpikeID SbsBaseLayer_generateSpikeIP(NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    NeuronState random_s = ((NeuronState)genrand()) / ((NeuronState)0xFFFFFFFF);
    NeuronState sum      = 0.0f;
    SpikeID     spikeID;

    ASSERT(random_s <= 1.0F);

    for (spikeID = 0; spikeID < size; spikeID ++)
    {
        sum += state_vector[spikeID];

        //ASSERT(sum <= 1 + 1e-5);

        if (random_s <= sum)
              return spikeID;
    }
  }

  return size - 1;
}

static void SbsBaseLayer_initialize(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->state_matrix != NULL);
  ASSERT(layer->state_matrix->data != NULL);

  if ((layer != NULL) && (layer->state_matrix != NULL) && (layer->state_matrix->data != NULL))
  {
    Multivector * state_matrix      = layer->state_matrix;
    uint16_t      rows              = state_matrix->dimension_size[0];
    uint16_t      columns           = state_matrix->dimension_size[1];
    uint16_t      neurons           = state_matrix->dimension_size[2];
    NeuronState * state_matrix_data = state_matrix->data;

    uint16_t row;
    uint16_t column;
    size_t   current_row_index;

    for (row = 0; row < rows; row++)
    {
      current_row_index = row * columns * neurons;
      for (column = 0; column < columns; column++)
      {
        SbsBaseLayer_initializeIP(&state_matrix_data[current_row_index + column * neurons], neurons);
      }
    }
  }
}

static void SbsBaseLayer_giveWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->weight_matrix = (Multivector *) weight_matrix;
}

static void SbsBaseLayer_setEpsilon(SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->epsilon = epsilon;
}

static Multivector * SbsBaseLayer_generateSpikes(SbsBaseLayer * layer)
{
  Multivector * spike_matrix = NULL;
  ASSERT(layer != NULL);
  ASSERT(layer->state_matrix != NULL);
  ASSERT(layer->spike_matrix != NULL);
  ASSERT(layer->state_matrix->data != NULL);
  ASSERT(layer->spike_matrix->data != NULL);

  if (   (layer != NULL)
      && (layer->state_matrix != NULL)
      && (layer->spike_matrix != NULL)
      && (layer->state_matrix->data != NULL)
      && (layer->spike_matrix->data != NULL))
  {
      Multivector * state_matrix      = layer->state_matrix;
      uint16_t      rows              = state_matrix->dimension_size[0];
      uint16_t      columns           = state_matrix->dimension_size[1];
      uint16_t      neurons           = state_matrix->dimension_size[2];
      NeuronState * state_matrix_data = state_matrix->data;
      SpikeID *     spike_matrix_data = layer->spike_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t   current_row_index;
      size_t   current_row_column_index;

      for (row = 0; row < rows; row++)
      {
        current_row_index = columns * row;
        for (column = 0; column < columns; column++)
        {
            current_row_column_index = current_row_index + column;
            spike_matrix_data[current_row_column_index] = SbsBaseLayer_generateSpikeIP(&state_matrix_data[current_row_column_index * neurons], neurons);
        }
      }

      spike_matrix = layer->spike_matrix;
  }

  return spike_matrix;
}

static void SbsBaseLayer_update(SbsBaseLayer * layer, Multivector * input_spike_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(layer->state_matrix != NULL);
  ASSERT(layer->state_matrix->data != NULL);
  ASSERT(layer->weight_matrix != NULL);
  ASSERT(layer->weight_matrix->data != NULL);

  ASSERT(0 < layer->kernel_size);

  ASSERT(input_spike_matrix != NULL);
  ASSERT(input_spike_matrix->data != NULL);

  if (   (layer != NULL)
      && (layer->state_matrix != NULL)
      && (layer->state_matrix->data != NULL)
      && (layer->weight_matrix != NULL)
      && (layer->weight_matrix->data != NULL)
      && (input_spike_matrix != NULL)
      && (input_spike_matrix->data != NULL))
  {
      SpikeID   spikeID       = 0;
      SpikeID * spike_data    = input_spike_matrix->data;
      uint16_t  spike_rows    = input_spike_matrix->dimension_size[0];
      uint16_t  spike_columns = input_spike_matrix->dimension_size[1];

      NeuronState * weight_data    = layer->weight_matrix->data;
      NeuronState * weight_vector  = NULL;
      uint16_t      weight_columns = layer->weight_matrix->dimension_size[1];

      NeuronState * state_data     = layer->state_matrix->data;
      NeuronState * state_vector   = NULL;
      uint16_t      state_row_size = layer->state_matrix->dimension_size[1] * layer->state_matrix->dimension_size[2];
      uint16_t      neurons        = layer->state_matrix->dimension_size[2];

      uint16_t kernel_stride  = layer->kernel_stride;
      uint16_t kernel_size    = layer->kernel_size;
      uint16_t row_shift      = kernel_size;
      uint16_t column_shift   = 1;
      uint16_t section_shift  = 0;


      uint16_t layer_row;         /* Row index for navigation on the layer */
      uint16_t layer_column;      /* Column index for navigation on the layer */
      uint16_t kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
      uint16_t kernel_row_pos;    /* Kernel row position for navigation on the spike matrix */
      uint16_t kernel_row;        /* Row index for navigation inside kernel */
      uint16_t kernel_column;     /* Column index for navigation inside kernel */

      uint16_t  spike_row_index;

      uint16_t neurons_previous_Layer = layer->neurons_previous_Layer;

      float epsilon = layer->epsilon;

      ASSERT(weight_columns == neurons);

      if (weight_columns != neurons)
        return;

      if (layer->weight_shift == ROW_SHIFT)
      {
        row_shift = 1;
        column_shift = kernel_size;
      }

      /* Update begins */
      for (kernel_row_pos = 0, layer_row = 0;
           kernel_row_pos < spike_rows - (kernel_size - 1);
           kernel_row_pos += kernel_stride, layer_row ++)
      {
        for (kernel_column_pos = 0, layer_column = 0;
             kernel_column_pos < spike_columns - (kernel_size - 1);
             kernel_column_pos += kernel_stride, layer_column ++)
        {
          state_vector = &state_data[layer_row * state_row_size + layer_column * neurons];
          for (kernel_row = 0; kernel_row < kernel_size; kernel_row ++)
          {
              spike_row_index = (kernel_row_pos + kernel_row) * spike_columns;
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column ++)
            {
              spikeID = spike_data[spike_row_index + kernel_column_pos + kernel_column];

              section_shift = (kernel_row * row_shift + kernel_column * column_shift) * neurons_previous_Layer;

              weight_vector = &weight_data[(spikeID + section_shift) * weight_columns];

              SbsBaseLayer_updateIP(layer, state_vector, weight_vector, neurons, epsilon);
            }
          }
        }
      }
      /* Update ends*/
  }
}

/*****************************************************************************/

static SbsNetwork * SbsBaseNetwork_new(void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc(sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
      memset(network, 0x0, sizeof(SbsBaseNetwork));
      network->vtbl = _SbsNetwork;
      network->input_label = (uint8_t)-1;
      network->inferred_output = (uint8_t)-1;

      sgenrand(666); /*TODO: Create MT19937 object wrapper */
  }

  ASSERT(network->size == 0);
  ASSERT(network->layer_array == NULL);

  return (SbsNetwork *) network;
}

static void SbsBaseNetwork_delete(SbsNetwork ** network_ptr)
{
  ASSERT(network_ptr != NULL);
  ASSERT(*network_ptr != NULL);

  if ((network_ptr != NULL) && (*network_ptr != NULL))
  {
    SbsBaseNetwork ** network = (SbsBaseNetwork **) network_ptr;
    while (0 < (*network)->size)
      SbsBaseLayer_delete((SbsLayer **)&(*network)->layer_array[--((*network)->size)]);

    free(*network);
    *network = NULL;
  }
}

static void SbsBaseNetwork_giveLayer(SbsNetwork * network_ptr, SbsLayer * layer)
{
  ASSERT(network_ptr != NULL);
  ASSERT(layer != NULL);

  if ((network_ptr != NULL) && (layer != NULL))
  {
    SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
    SbsBaseLayer ** layer_array = network->layer_array;
    uint8_t size = network->size;

    ASSERT(size < 0xFF);

    layer_array = realloc(layer_array, (size + 1) * sizeof(SbsBaseLayer *));

    ASSERT(layer_array != NULL);

    if (layer_array != NULL)
    {
        layer_array[size] = (SbsBaseLayer *)layer;

        network->layer_array = layer_array;
        network->size ++;
    }
  }
}

static void SbsBaseNetwork_loadInput(SbsNetwork * network_ptr, char * file_name)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(1 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(*network->layer_array != NULL);

  ASSERT(file_name != NULL);

  if ((network != NULL)
      && (1 <= network->size)
      && (network->layer_array != NULL) && (*network->layer_array != NULL)
      && (file_name != NULL))
  {
#ifdef USE_XILINX
    FIL fil; /* File object */
    FRESULT rc;
    rc = f_open (&fil, file_name, FA_READ);
    ASSERT(rc == FR_OK);

    if (rc == FR_OK)
    {
      SbsBaseLayer * input_layer = network->layer_array[0];
      uint16_t       rows        = input_layer->state_matrix->dimension_size[0];
      uint16_t       columns     = input_layer->state_matrix->dimension_size[1];
      uint16_t       neurons     = input_layer->state_matrix->dimension_size[2];
      NeuronState *  data        = input_layer->state_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t   read_result = 0;

      uint8_t good_reading_flag = 1;

      size_t inference_population_size = sizeof(NeuronState) * neurons;

      for (column = 0; (column < columns) && good_reading_flag; column++)
        for (row = 0; (row < rows) && good_reading_flag; row++)
        {
          rc = f_read (&fil, &data[column * neurons + row * columns * neurons],
                       inference_population_size, &read_result);

          good_reading_flag = read_result == inference_population_size;
        }

      if (good_reading_flag)
      {
        rc = f_read (&fil, &network->input_label, sizeof(uint8_t), &read_result);
        network->input_label--;
        good_reading_flag = read_result == sizeof(uint8_t);
      }

      f_close (&fil);
      ASSERT(good_reading_flag);
    }
#else
    FILE * file = fopen(file_name, "rb");

    ASSERT(file != NULL);

    if (file != NULL)
    {
      SbsBaseLayer * input_layer = network->layer_array[0];
      uint16_t rows = input_layer->state_matrix->dimension_size[0];
      uint16_t columns = input_layer->state_matrix->dimension_size[1];
      uint16_t neurons = input_layer->state_matrix->dimension_size[2];
      NeuronState * data = input_layer->state_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t read_result = 0;

      uint8_t good_reading_flag = 1;

      size_t inference_population_size = sizeof(NeuronState) * neurons;

      for (column = 0; (column < columns) && good_reading_flag; column++)
        for (row = 0; (row < rows) && good_reading_flag; row++)
        {
          read_result = fread (&data[column * neurons + row * columns * neurons], 1,
              inference_population_size, file);

          good_reading_flag = read_result == inference_population_size;
        }

      if (good_reading_flag)
      {
        read_result = fread(&network->input_label, 1, sizeof(uint8_t), file);
        network->input_label --;
        good_reading_flag = read_result == sizeof(uint8_t);
      }

      fclose(file);
      ASSERT(good_reading_flag);
    }
#endif
  }
}

static void SbsBaseNetwork_updateCycle(SbsNetwork * network_ptr, uint16_t cycles)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  uint16_t cycle;
  char file_name[80];
  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    uint16_t i;
    /* Initialize all layers except the input-layer */
    for (i = 1; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
    }

    /************************ Begins Update cycle **************************/
    for (cycle = 0; cycle < cycles; cycle ++)
    {
      for (i = 0; i < network->size; i++)
      {
        if (i < network->size - 1)
        {
          SbsBaseLayer_generateSpikes(network->layer_array[i]);

#if defined(SAVE_SPIKES)
          sprintf (file_name, "spike_layer[%d]_cycle[%d].csv", i, cycle);
          Multivector_saveToCSV (network->layer_array[i]->spike_matrix, file_name);
#endif
        }

        if (0 < i)
          SbsBaseLayer_update(network->layer_array[i],
              network->layer_array[i - 1]->spike_matrix);
      }

      if (cycle % 100 == 0)
        printf(" - Spike cycle: %d\n", cycle);
    }
    /************************ Ends Update cycle ****************************/

    /************************ Get inferred output **************************/
    {
      NeuronState max_value = 0;
      SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
      Multivector * output_state_matrix = output_layer->state_matrix;
      NeuronState * output_state_vector = output_state_matrix->data;

      ASSERT(output_state_matrix->dimensionality == 3);
      ASSERT(output_state_matrix->dimension_size[0] == 1);
      ASSERT(output_state_matrix->dimension_size[1] == 1);
      ASSERT(0 < output_state_matrix->dimension_size[2]);

      for (i = 0; i < output_state_matrix->dimension_size[2]; i++)
      {
        NeuronState h = output_state_vector[i]; /* Ensure data alignment */
        if (max_value < h)
        {
          network->inferred_output = i;
          max_value = h;
        }
      }
    }
  }
}

static uint8_t SbsBaseNetwork_getInferredOutput(SbsNetwork * network)
{
  uint8_t inferred_output = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    inferred_output = ((SbsBaseNetwork *) network)->inferred_output;
  }

  return inferred_output;
}

static uint8_t SbsBaseNetwork_getInputLabel(SbsNetwork * network)
{
  uint8_t input_label = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    input_label = ((SbsBaseNetwork *) network)->input_label;
  }

  return input_label;
}

static void SbsBaseNetwork_getOutputVector(SbsNetwork * network_ptr, NeuronState ** output_vector, uint16_t * output_vector_size)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(0 < network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(network->layer_array[network->size - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != NULL);

  if ((network != NULL)
      && (0 < network->size)
      && (network->layer_array != NULL)
      && (network->layer_array != NULL)
      && (output_vector != NULL)
      && (output_vector_size != NULL))
  {
    SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
    Multivector * output_state_matrix = output_layer->state_matrix;

    ASSERT(output_state_matrix->data != NULL);
    ASSERT(output_state_matrix->dimensionality == 3);
    ASSERT(output_state_matrix->dimension_size[0] == 1);
    ASSERT(output_state_matrix->dimension_size[1] == 1);
    ASSERT(0 < output_state_matrix->dimension_size[2]);

    * output_vector = output_state_matrix->data;
    * output_vector_size = output_state_matrix->dimension_size[2];
  }
}

static size_t SbsBaseNetwork_getMemorySize(SbsNetwork * network)
{
  return Memory_getBlockSize();
}
/*****************************************************************************/

static SbsLayer * SbsInputLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new(rows, columns, neurons, 0, 0, ROW_SHIFT, 0);
}

static SbsLayer * SbsConvolutionLayer_new(uint16_t rows,
                                            uint16_t columns,
                                            uint16_t neurons,
                                            uint16_t kernel_size,
                                            WeightShift weight_shift,
                                            uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (rows, columns, neurons, kernel_size, 1, weight_shift,
                           neurons_prev_Layer);
}

static SbsLayer * SbsPoolingLayer_new(uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift,
                                    uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (rows, columns, neurons, kernel_size,
                                        kernel_size, weight_shift, neurons_prev_Layer);
}

static SbsLayer * SbsFullyConnectedLayer_new(uint16_t neurons,
                                                  uint16_t kernel_size,
                                                  WeightShift weight_shift,
                                                  uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (1, 1, neurons, kernel_size, 1,
                                        weight_shift, neurons_prev_Layer);
}

static SbsLayer * SbsOutputLayer_new(uint16_t neurons,
                                  WeightShift weight_shift,
                                  uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (1, 1, neurons, 1, 1, weight_shift,
                                        neurons_prev_Layer);
}
/*****************************************************************************/

static SbsWeightMatrix SbsWeightMatrix_new(uint16_t rows, uint16_t columns, char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new(sizeof(Weight), 2, rows, columns);

    ASSERT(weight_watrix != NULL);
    ASSERT(weight_watrix->dimensionality == 2);
    ASSERT(weight_watrix->data != NULL);
    ASSERT(weight_watrix->dimension_size[0] == rows);
    ASSERT(weight_watrix->dimension_size[1] == columns);

    if ((weight_watrix != NULL)
        && (weight_watrix->dimensionality == 2)
        && (weight_watrix->data != NULL)
        && (weight_watrix->dimension_size[0] == rows)
        && (weight_watrix->dimension_size[1] == columns))
    {
#ifdef USE_XILINX
      FIL fil; /* File object */
      FRESULT rc;
      rc = f_open (&fil, file_name, FA_READ);
      ASSERT(rc == FR_OK);

      if (rc == FR_OK)
      {
        size_t read_size;
        size_t data_size = rows * columns * sizeof(Weight);
        rc = f_read (&fil, weight_watrix->data, data_size, &read_size);
        ASSERT((rc == FR_OK) && (read_size == data_size));
        f_close (&fil);
      }
      else Multivector_delete (&weight_watrix);
#else
      FILE * file = fopen(file_name, "rb");

      ASSERT(file != NULL);

      if (file != NULL)
      {
        size_t data_size = rows * columns * sizeof(Weight);
        size_t read_result = fread(weight_watrix->data, 1, data_size, file);
        ASSERT(data_size == read_result);
        fclose(file);
      }
      else
        Multivector_delete(&weight_watrix);
#endif
    }
  }

  return weight_watrix;
}

/*****************************************************************************/

SbsNetwork _SbsNetwork = {SbsBaseNetwork_new,
                          SbsBaseNetwork_delete,
                          SbsBaseNetwork_giveLayer,
                          SbsBaseNetwork_loadInput,
                          SbsBaseNetwork_updateCycle,
                          SbsBaseNetwork_getInferredOutput,
                          SbsBaseNetwork_getInputLabel,
                          SbsBaseNetwork_getOutputVector,
                          SbsBaseNetwork_getMemorySize};

SbsLayer _SbsLayer = {SbsBaseLayer_new,
                      SbsBaseLayer_delete,
                      SbsBaseLayer_setEpsilon,
                      SbsBaseLayer_giveWeights};

SbsNew sbs_new = {SbsBaseNetwork_new,
                  SbsBaseLayer_new,
                  SbsWeightMatrix_new,
                  SbsInputLayer_new,
                  SbsConvolutionLayer_new,
                  SbsPoolingLayer_new,
                  SbsFullyConnectedLayer_new,
                  SbsOutputLayer_new};


/*****************************************************************************/
