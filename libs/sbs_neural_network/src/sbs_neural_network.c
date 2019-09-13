/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: yarib
 */


#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "stddef.h"
#include "stdarg.h"

#include "sbs_neural_network.h"
#include "mt19937int.h"

#define ASSERT(expr)  assert(expr)

/*****************************************************************************/
#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */


typedef float     Weight;
typedef uint16_t  SpikeID;

typedef struct
{
  void *   data;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
} Multivector;

typedef struct
{
  Multivector * state_matrix;
  Multivector * weight_matrix;
  Multivector * spike_matrix;

  uint16_t    kernel_size;
  uint16_t    kernel_stride;
  uint16_t    neurons_previous_Layer;
  WeightShift weight_shift;
  float       epsilon;
} SbsBaseLayer;

typedef struct
{
  struct SbsNetwork_VTable vtbl;

  uint8_t           size;
  SbsBaseLayer **   layer_array;
  uint8_t           input_label;
  uint8_t           inferred_output;
} SbsBaseNetwork;


#pragma pack(pop)   /* restore original alignment from stack */

/*****************************************************************************/

static void * ram_block_request(size_t size)
{
  static size_t pool_size = 0;
  pool_size += size;

  return malloc(size);
}

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

      multivector->data = ram_block_request(data_size * data_type_size);

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size * data_type_size);

      multivector->dimensionality = dimensionality;
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
/*****************************************************************************/
/*****************************************************************************/

static SbsBaseLayer * SbsBaseLayer_new(uint16_t rows,
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

    /* Assign parameters */
    layer->kernel_size   = kernel_size;
    layer->kernel_stride = kernel_stride;
    layer->weight_shift  = weight_shift;
    layer->neurons_previous_Layer = neurons_previous_Layer;
  }

  return layer;
}

static void SbsBaseLayer_delete(SbsBaseLayer ** layer)
{
  ASSERT(layer!= NULL);
  ASSERT(*layer!= NULL);
  if ((layer!= NULL) && (*layer!= NULL))
  {
    Multivector_delete(&((*layer)->state_matrix));
    Multivector_delete(&((*layer)->spike_matrix));
    if ((*layer)->weight_matrix != NULL) Multivector_delete(&((*layer)->weight_matrix));
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

static void SbsBaseLayer_updateIP(NeuronState * state_vector, Weight * weight_vector, uint16_t size, float epsilon)
{
  ASSERT(state_vector != NULL);
  ASSERT(weight_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (weight_vector != NULL) && (0 < size))
  {
    static NeuronState * temp_data      = NULL;
    static size_t        temp_data_size = 0;

    NeuronState sum             = 0.0f;
    NeuronState reverse_epsilon = 1.0f / (1.0f + epsilon);
    NeuronState epsion_over_sum = 0.0f;
    uint16_t    neuron;

    if (temp_data_size < size)
    {
      temp_data = (NeuronState *) realloc(temp_data, size * sizeof(NeuronState));

      ASSERT(temp_data != NULL);
      if (temp_data == NULL) return;

      temp_data_size = size;
    }

    for (neuron = 0; neuron < size; neuron ++)
    {
      temp_data[neuron] = state_vector[neuron] * weight_vector[neuron];
      sum += temp_data[neuron];
    }

    if (sum < 1e-20) // TODO: DEFINE constant
      return;

    epsion_over_sum = epsilon / sum;

    for (neuron = 0; neuron < size; neuron ++)
      state_vector[neuron] = reverse_epsilon * (state_vector[neuron] + temp_data[neuron] * epsion_over_sum);
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

        ASSERT(sum <= 1 + 1e-5);

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

void SbsBaseLayer_giveWeights(void * layer, void * weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->weight_matrix = weight_matrix;
}

void SbsBaseLayer_setEpsilon(void * layer, float epsilon)
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

              SbsBaseLayer_updateIP(state_vector, weight_vector, neurons, epsilon);
            }
          }
        }
      }
      /* Update ends*/
  }
}

/*****************************************************************************/

static SbsBaseNetwork * SbsBaseNetwork_new(void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc(sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
      memset(network, 0x0, sizeof(SbsBaseNetwork));
      network->vtbl = SbsNetwork_vtable;
      network->input_label = (uint8_t)-1;
      network->inferred_output = (uint8_t)-1;
  }

  ASSERT(network->size == 0);
  ASSERT(network->layer_array == NULL);

  return network;
}

static void SbsBaseNetwork_delete(SbsBaseNetwork ** network)
{
  ASSERT(network != NULL);
  ASSERT(*network != NULL);

  if ((network != NULL) && (*network != NULL))
  {
    while (0 < (*network)->size)
      SbsBaseLayer_delete(&(*network)->layer_array[--((*network)->size)]);

    free(*network);
    *network = NULL;
  }
}

static void SbsBaseNetwork_giveLayer(SbsBaseNetwork * network, SbsBaseLayer * layer)
{
  ASSERT(network != NULL);
  ASSERT(layer != NULL);

  if ((network != NULL) && (layer != NULL))
  {
    SbsBaseLayer ** layer_array = network->layer_array;
    uint8_t size = network->size;

    ASSERT(size < 0xFF);

    layer_array = realloc(layer_array, (size + 1) * sizeof(SbsBaseLayer *));

    ASSERT(layer_array != NULL);

    if (layer_array != NULL)
    {
        layer_array[size] = layer;

        network->layer_array = layer_array;
        network->size ++;
    }
  }
}

static void SbsBaseNetwork_loadInput(SbsBaseNetwork * network, char * file_name)
{
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
  }
}

static void SbsBaseNetwork_updateCycle(SbsBaseNetwork * network, uint16_t cycles)
{
  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    uint16_t i;
    /* Initialize all layers except the input-layer */
    for (i = 1; i < network->size; i ++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
    }

    Multivector ** spike_array = malloc ((network->size - 1) * sizeof(Multivector *));

    ASSERT(spike_array != NULL);

    if (spike_array != NULL)
    {
      /************************ Begins Update cycle **************************/
      while (cycles --)
      {
        for (i = 0; i < network->size - 1; i ++)
        {
          spike_array[i] = SbsBaseLayer_generateSpikes(network->layer_array[i]);
        }

        for (i = 1; i < network->size; i ++)
        {
          SbsBaseLayer_update(network->layer_array[i], spike_array[i - 1]);
        }

        if (cycles % 100 == 0) printf(" - Spike cycle: %d\n", cycles);
      }
      /************************ Ends Update cycle ****************************/

      free(spike_array);

      /************************ Update inferred output ************************/
      {
        NeuronState max_value = 0;
        uint16_t max_value_position = 0;
        SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
        Multivector * output_state_matrix = output_layer->state_matrix;
        NeuronState * output_state_vector = output_state_matrix->data;

        ASSERT(output_state_matrix->dimensionality == 3);
        ASSERT(output_state_matrix->dimension_size[0] == 1);
        ASSERT(output_state_matrix->dimension_size[1] == 1);
        ASSERT(0 < output_state_matrix->dimension_size[2]);

        for (i = 0; i < output_state_matrix->dimension_size[2]; i++)
        {
          if (max_value < output_state_vector[i])
          {
            max_value_position = i;
            max_value = output_state_vector[i];
          }
        }

        network->inferred_output = max_value_position;
      }
    }
  }
}

static uint8_t SbsBaseNetwork_getInferredOutput(SbsBaseNetwork * network)
{
  uint8_t inferred_output = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    inferred_output = network->inferred_output;
  }

  return inferred_output;
}

static uint8_t SbsBaseNetwork_getInputLabel(SbsBaseNetwork * network)
{
  uint8_t input_label = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    input_label = network->input_label;
  }

  return input_label;
}

static void SbsBaseNetwork_getOutputVector(SbsBaseNetwork * network, NeuronState ** output_vector, uint16_t * output_vector_size)
{
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

/*****************************************************************************/

SbsInputLayer SbsInputLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons)
{
  return SbsBaseLayer_new(rows, columns, neurons, 0, 0, ROW_SHIFT, 0);
}

SbsConvolutionLayer SbsConvolutionLayer_new(uint16_t rows,
                                            uint16_t columns,
                                            uint16_t neurons,
                                            uint16_t kernel_size,
                                            WeightShift weight_shift,
                                            uint16_t neurons_prev_Layer)
{
  return SbsBaseLayer_new (rows, columns, neurons, kernel_size, 1, weight_shift,
                           neurons_prev_Layer);
}

SbsPoolingLayer SbsPoolingLayer_new(uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift,
                                    uint16_t neurons_prev_Layer)
{
  return SbsBaseLayer_new (rows, columns, neurons, kernel_size, kernel_size,
                           weight_shift, neurons_prev_Layer);
}

SbsFullyConnectedLayer SbsFullyConnectedLayer_new(uint16_t neurons,
                                                  uint16_t kernel_size,
                                                  WeightShift weight_shift,
                                                  uint16_t neurons_prev_Layer)
{
  return SbsBaseLayer_new(1, 1, neurons, kernel_size, 1, weight_shift, neurons_prev_Layer);
}

SbsOutputLayer SbsOutputLayer_new(uint16_t neurons,
                                  WeightShift weight_shift,
                                  uint16_t neurons_prev_Layer)
{
  return SbsBaseLayer_new(1, 1, neurons, 1, 1, weight_shift, neurons_prev_Layer);
}
/*****************************************************************************/

SbsWeightMatrix SbsWeightMatrix_new(uint16_t rows, uint16_t columns, char * file_name)
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
    }
  }

  return weight_watrix;
}

/*****************************************************************************/

struct SbsNetwork_VTable SbsNetwork_vtable = {SbsBaseNetwork_new,
                                              SbsBaseNetwork_delete,
                                              SbsBaseNetwork_giveLayer,
                                              SbsBaseNetwork_loadInput,
                                              SbsBaseNetwork_updateCycle,
                                              SbsBaseNetwork_getInferredOutput,
                                              SbsBaseNetwork_getInputLabel,
                                              SbsBaseNetwork_getOutputVector};

//struct SbsWeightMatrix_VTable SbsWeightMatrix_vtable = {SbsWeightMatrix_new};

void sbs_test(void)
{
//  NeuronState * output_vector;
//  uint16_t output_vector_size;
//
//  sgenrand(666);
//
//  /*********************/
//  // ********** Create SBS Neural Network **********
//  printf("\n==========  SbS Neural Network  ===============\n");
//  printf("\n==========  MNIST example  ====================\n");
//
//  SbsNetwork * network = SbsNetwork_new();
//
//  // Instantiate SBS Network objects
//  SbsInputLayer input_layer = SbsInputLayer_new(24, 24, 50);
//  SbsNetwork_giveLayer(network, input_layer);
//
//  SbsWeightMatrix P_IN_H1 = SbsWeightMatrix_new(2 * 5 * 5, 32, "/home/nevarez/Downloads/MNIST/W_X_H1_Iter0.bin");
//
//  SbsConvolutionLayer H1 = SbsConvolutionLayer_new(24, 24, 32, 1, ROW_SHIFT, 50);
//  SbsBaseLayer_setEpsilon(H1, 0.1);
//  SbsBaseLayer_giveWeights(H1, P_IN_H1);
//  SbsNetwork_giveLayer(network, H1);
//
//  SbsWeightMatrix P_H1_H2 = SbsWeightMatrix_new(32 * 2 * 2, 32, "/home/nevarez/Downloads/MNIST/W_H1_H2.bin");
//
//  SbsPoolingLayer H2 = SbsPoolingLayer_new(12, 12, 32, 2, COLUMN_SHIFT, 32);
//  SbsBaseLayer_setEpsilon(H2, 0.1 / 4.0);
//  SbsBaseLayer_giveWeights(H2, P_H1_H2);
//  SbsNetwork_giveLayer(network, H2);
//
//  SbsWeightMatrix P_H2_H3 = SbsWeightMatrix_new(32 * 5 * 5, 64, "/home/nevarez/Downloads/MNIST/W_H2_H3_Iter0.bin");
//
//  SbsConvolutionLayer H3 = SbsConvolutionLayer_new(8, 8, 64, 5, COLUMN_SHIFT, 32);
//  SbsBaseLayer_setEpsilon(H3, 0.1 / 25.0);
//  SbsBaseLayer_giveWeights(H3, P_H2_H3);
//  SbsNetwork_giveLayer(network, H3);
//
//  SbsWeightMatrix P_H3_H4 = SbsWeightMatrix_new(64 * 2 * 2, 64, "/home/nevarez/Downloads/MNIST/W_H3_H4.bin");
//
//  SbsPoolingLayer H4 = SbsPoolingLayer_new(4, 4, 64, 2, COLUMN_SHIFT, 64);
//  SbsBaseLayer_setEpsilon(H4, 0.1 / 4.0);
//  SbsBaseLayer_giveWeights(H4, P_H3_H4);
//  SbsNetwork_giveLayer(network, H4);
//
//  SbsWeightMatrix P_H4_H5 = SbsWeightMatrix_new(64 * 4 * 4, 1024, "/home/nevarez/Downloads/MNIST/W_H4_H5_Iter0.bin");
//
//  SbsFullyConnectedLayer H5 = SbsFullyConnectedLayer_new(1024, 4, ROW_SHIFT, 64);
//  SbsBaseLayer_setEpsilon(H5, 0.1 / 16.0);
//  SbsBaseLayer_giveWeights(H5, P_H4_H5);
//  SbsNetwork_giveLayer(network, H5);
//
//  SbsWeightMatrix P_H5_HY = SbsWeightMatrix_new(1024, 10, "/home/nevarez/Downloads/MNIST/W_H5_HY_Iter0.bin");
//
//  SbsOutputLayer HY = SbsOutputLayer_new(10, ROW_SHIFT, 0);
//  SbsBaseLayer_setEpsilon(HY, 0.1);
//  SbsBaseLayer_giveWeights(HY, P_H5_HY);
//  SbsNetwork_giveLayer(network, HY);
//
//    // Perform Network load pattern and update cycle
//  SbsNetwork_loadInput(network, "/home/nevarez/Downloads/MNIST/Pattern/Input_1.bin");
//  SbsNetwork_updateCycle(network, 1000);
//
//  printf("\n==========  Results ===========================\n");
//
//  printf("\n Output value: %d \n", SbsNetwork_getInferredOutput(network));
//  printf("\n Label value: %d \n", SbsNetwork_getInputLabel(network));
//
//  SbsNetwork_getOutputVector(network, &output_vector, &output_vector_size);
//
//  printf("\n==========  Output layer values ===============\n");
//
//  while (output_vector_size --)
//  {
//    printf("[ %d ] = %.6f\n", output_vector_size, output_vector[output_vector_size]);
//  }
//
//  SbsNetwork_delete(&network);
//  /*********************/
}

/*****************************************************************************/
