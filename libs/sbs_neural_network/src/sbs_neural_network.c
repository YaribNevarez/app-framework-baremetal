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


typedef float     NeuronState;
typedef float     Weight;
typedef uint16_t  SpikeID;

typedef struct
{
  void *   data;
  uint8_t  data_type_size;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
} Multivector;

typedef enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
} WeightShift;

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
  uint8_t         size;
  SbsBaseLayer ** layer_array;
} SbsNetwork;

/*****************************************************************************/

void * ram_block_request(size_t size)
{
  static size_t pool_size = 0;
  pool_size += size;

  return malloc(size);
}

/*****************************************************************************/

Multivector * Multivector_new(uint8_t data_type_size, uint8_t dimensionality, ...)
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
        data_size *= (multivector->dimension_size[arg] = (uint8_t) va_arg(argument_list, int));

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

void Multivector_delete(Multivector ** multivector)
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

SbsBaseLayer * SbsBaseLayer_new(uint16_t rows,
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

void SbsBaseLayer_delete(SbsBaseLayer ** layer)
{
  ASSERT(layer!= NULL);
  ASSERT(*layer!= NULL);
  if ((layer!= NULL) && (*layer!= NULL))
  {
    Multivector_delete(&((*layer)->state_matrix));
    Multivector_delete(&((*layer)->spike_matrix));
    free(*layer);
    *layer = NULL;
  }
}


void SbsBaseLayer_initializeIP(NeuronState * state_vector, uint16_t size)
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

void SbsBaseLayer_updateIP(NeuronState * state_vector, Weight * weight_vector, uint16_t size, float epsilon)
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

SpikeID SbsBaseLayer_generateSpikeIP(NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    NeuronState random_s = genrand() / 0xFFFFFFFF;
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

void SbsBaseLayer_initialize(SbsBaseLayer * layer)
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
    size_t   current_row_column_index;

    for (row = 0; row < rows; row++)
    {
      current_row_index = columns * row;
      for (column = 0; column < columns; column++)
      {
        current_row_column_index = current_row_index + column;
        SbsBaseLayer_initializeIP(&state_matrix_data[current_row_column_index * neurons], neurons);
      }
    }
  }
}

void SbsBaseLayer_setWeights(SbsBaseLayer * layer, Multivector * weight_matrix)
{
  ASSERT(layer != NULL);
  /*ASSERT(weight_matrix != NULL);*/ /* NULL is allowed? */

  if (layer != NULL)
    layer->weight_matrix = weight_matrix;
}

void SbsBaseLayer_setEpsilon(SbsBaseLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    layer->epsilon = epsilon;
}

Multivector * SbsBaseLayer_generateSpikes(SbsBaseLayer * layer)
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

void SbsBaseLayer_update(SbsBaseLayer * layer, Multivector * input_spike_matrix)
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

SbsNetwork * SbsNetwork_new(void)
{
  SbsNetwork * network = NULL;

  network = malloc(sizeof(SbsNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
      memset(network, 0x0, sizeof(SbsNetwork));
  }

  ASSERT(network->size == 0);
  ASSERT(network->layer_array == NULL);

  return network;
}

void SbsNetwork_delete(SbsNetwork ** network)
{
  ASSERT(network != NULL);
  ASSERT(*network != NULL);

  if ((network != NULL) && (*network != NULL))
  {
    free(*network);
    *network = NULL;
  }
}

void SbsNetwork_addLayer(SbsNetwork * network, SbsBaseLayer * layer)
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

void SbsNetwork_updateCycle(SbsNetwork * network, uint16_t cycles)
{
  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    uint8_t i;
    /* Initialize all layers except the input-layer */
    for (i = 1; i < network->size; i ++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
    }

    Multivector ** spike_array = malloc((network->size - 1)* sizeof(Multivector *));

    ASSERT(spike_array != NULL);

    if (spike_array != NULL)
    {
      while (cycles --)
      {
        for (i = 0; i < network->size - 1; i ++)
        {
          spike_array[i] = SbsBaseLayer_generateSpikes(network->layer_array[i]);
        }

        for (i = 1; i < network->size; i ++)
        {
          SbsBaseLayer_update(network->layer_array[i], spike_array[i]);
        }

        if (cycles % 100 == 0) printf(" - Spike cycle: %d", cycles);
      }

      free(spike_array);
    }
  }
}

/*****************************************************************************/

void sbs_test(void)
{
  sgenrand(666);
  //Multivector * network = SbsNetwork_new();

  //Multivector * input_layer = SbsLayer_new();
}

/*****************************************************************************/
