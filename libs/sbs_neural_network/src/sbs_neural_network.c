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

typedef void * SbsInputLayer;
typedef void * SbsConvolutionLayer;
typedef void * SbsPoolingLayer;
typedef void * SbsFullyConnectedLayer;
typedef void * SbsOutputLayer;
typedef void * SbsWeightMatrix;

typedef struct
{
  uint8_t         size;
  SbsBaseLayer ** layer_array;
} SbsNetwork;


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

static void SbsBaseLayer_setWeights(SbsBaseLayer * layer, Multivector * weight_matrix)
{
  ASSERT(layer != NULL);
  /*ASSERT(weight_matrix != NULL);*/ /* NULL is allowed? */

  if (layer != NULL)
    layer->weight_matrix = weight_matrix;
}

static void SbsBaseLayer_setEpsilon(SbsBaseLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    layer->epsilon = epsilon;
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

static SbsNetwork * SbsNetwork_new(void)
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

static void SbsNetwork_delete(SbsNetwork ** network)
{
  ASSERT(network != NULL);
  ASSERT(*network != NULL);

  if ((network != NULL) && (*network != NULL))
  {
    free(*network);
    *network = NULL;
  }
}

static void SbsNetwork_addLayer(SbsNetwork * network, SbsBaseLayer * layer)
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

static void SbsNetwork_updateCycle(SbsNetwork * network, uint16_t cycles)
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
  Multivector * weight_watrix = Multivector_new(sizeof(Weight), 2, rows, columns);

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
  }

  return weight_watrix;
}

/*****************************************************************************/

void sbs_test(void)
{
  sgenrand(666);
  /*********************/
  // ********** Create SBS Neural Network **********
  printf("\n==========  SbS Neural Network  ===============/n");
  printf("\n==========  MNIST example  ====================/n");

  SbsNetwork * network = SbsNetwork_new();

  // Instantiate SBS Network objects
  SbsInputLayer input_layer = SbsInputLayer_new(24, 24, 50);
  SbsNetwork_addLayer(network, input_layer);

//  sbs::Weights P_IN_H1(2 * 5 * 5, 32, "/home/nevarez/Downloads/MNIST/W_X_H1_Iter0.bin");
//
//  sbs::ConvolutionLayer H1(24, 24, 32, 1, sbs::BaseLayer::WeightSectionShift::ROW_SHIFT, 50);
//  H1.setEpsilon(0.1);
//  H1.setWeights(&P_IN_H1);
//  network.push_back(&H1);
//
//  sbs::Weights P_H1_H2(32 * 2 * 2, 32, "/home/nevarez/Downloads/MNIST/W_H1_H2.bin");
//
//  sbs::PoolingLayer H2(12, 12, 32, 2, sbs::BaseLayer::WeightSectionShift::COLUMN_SHIFT, 32);
//  H2.setEpsilon(0.1 / 4.0);
//  H2.setWeights(&P_H1_H2);
//  network.push_back(&H2);
//
//  sbs::Weights P_H2_H3(32 * 5 * 5, 64, "/home/nevarez/Downloads/MNIST/W_H2_H3_Iter0.bin");
//
//  sbs::ConvolutionLayer H3(8, 8, 64, 5, sbs::BaseLayer::WeightSectionShift::COLUMN_SHIFT, 32);
//  H3.setEpsilon(0.1 / 25.0);
//  H3.setWeights(&P_H2_H3);
//  network.push_back(&H3);
//
//  sbs::Weights P_H3_H4(64 * 2 * 2, 64, "/home/nevarez/Downloads/MNIST/W_H3_H4.bin");
//
//  sbs::PoolingLayer H4(4, 4, 64, 2, sbs::BaseLayer::WeightSectionShift::COLUMN_SHIFT, 64);
//  H4.setEpsilon(0.1 / 4.0);
//  H4.setWeights(&P_H3_H4);
//  network.push_back(&H4);
//
//  sbs::Weights P_H4_H5(64 * 4 * 4, 1024, "/home/nevarez/Downloads/MNIST/W_H4_H5_Iter0.bin");
//
//  sbs::FullyConnectedLayer H5(1024, 4, sbs::BaseLayer::WeightSectionShift::ROW_SHIFT, 64);
//  H5.setEpsilon(0.1 / 16.0);
//  H5.setWeights(&P_H4_H5);
//  network.push_back(&H5);
//
//  sbs::Weights P_H5_HY(1024, 10, "/home/nevarez/Downloads/MNIST/W_H5_HY_Iter0.bin");
//
//  sbs::OutputLayer HY(10, sbs::BaseLayer::WeightSectionShift::ROW_SHIFT, 0);
//  HY.setEpsilon(0.1);
//  HY.setWeights(&P_H5_HY);
//  network.push_back(&HY);
//
//  // Perform Network load pattern and update cycle
//  network.loadInput("/home/nevarez/Downloads/MNIST/Pattern/Input_33.bin");
//  network.updateCycle(1000);
//
//  std::cout << "\n==========  Results ===========================" << std::endl;
//
//  std::cout << "\n Output value: " << network.getOutput() << std::endl;
//  std::cout << "\n Label value: " << (int) network.getInputLabel() << std::endl;
//
//  std::cout << "\n==========  Output layer values ===============" << std::endl;
//
//  for (uint16_t i = 0; i < 10; i++)
//  {
//    std::cout << " [ " << i << " ] " << HY[0][0][i] << std::endl;
//  }
  /*********************/
}

/*****************************************************************************/
