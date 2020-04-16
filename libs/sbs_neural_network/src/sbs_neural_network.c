/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: Yarib Nevarez
 */


//#define DEBUG


#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "stddef.h"
#include "stdarg.h"

#include "timer.h"
#include "miscellaneous.h"
#include "memory_manager.h"

#include "sbs_neural_network.h"
#include "mt19937int.h"


#include "ff.h"
#include "xparameters.h"

#include "xscugic.h"

#include "multivector.h"

/*****************************************************************************/
#define   MEMORY_SIZE         (4771384)
#define   MAX_LAYER_SIZE      (28*28)
#define   MAX_KERNEL_SIZE     (5*5)
#define   MAX_IP_VECTOR_SIZE  (1024)  // Inference population size
#define   MAX_NETWORK_SIZE    (7)     // MAX number of layers in a network

/*****************************************************************************/

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */


typedef uint8_t   Weight;
typedef uint32_t  Random32;
typedef uint16_t  SpikeID;


typedef struct
{
  Format state_matrix_format;
  Format weight_matrix_format;
  Format spike_matrix_format;
  Format learning_matrix_format;
  Format weight_matrix_format_file_system;
  Format input_matrix_format_file_system;
} SbsSettings;

typedef struct
{
  uint16_t      x_pos;
  uint16_t      y_pos;
  Multivector * state_matrix;
  Multivector * spike_matrix;
  Multivector * weight_matrix;
} SbsLayerPartition;

typedef struct
{
  SbsLearningRule learning_rule;

  Multivector *   omega_matrix;
  Multivector *   a_matrix;
  Multivector *   b_matrix;
  float *         reco_vector;
  float *         delat_vector;
  double *        b_vector;

  unsigned int    number_of_patterns;
  unsigned int    current_pattern;
  double          gama;
} SbsLearningData;

typedef struct
{
  SbsLayer              vtbl;
  SbsLayerType          layer_type;
  SbsLayerPartition **  partition_array;
  uint8_t               num_partitions;
  uint16_t              rows;
  uint16_t              columns;
  uint16_t              vector_size;
  uint16_t              kernel_size;
  uint16_t              kernel_stride;
  WeightShift           weight_shift;
  Epsilon               epsilon;
  Multivector *         spike_matrix;
  SbsLearningData       learning_data;
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
Format state_matrix_format;
Format weight_matrix_format;
Format spike_matrix_format;

static SbsSettings SbsSettings_ =
{
    .state_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(uint16_t),
        .mantissa_bitlength = 11
    },
    .weight_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(uint8_t),
        .mantissa_bitlength = 4
    },
    .spike_matrix_format =
    {
        .representation = FIXED_POINT,
        .size = sizeof(uint16_t),
        .mantissa_bitlength = 0
    },
    .learning_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(double),
        .mantissa_bitlength = 0
    },
    .weight_matrix_format_file_system =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    },
    .input_matrix_format_file_system =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    }
};


/*****************************************************************************/
static SbsLayerPartition * SbsLayerPartition_new (uint16_t x_pos,
                                                  uint16_t y_pos,
                                                  uint16_t rows,
                                                  uint16_t columns,
                                                  uint16_t vector_size,
                                                  MemoryBlock * memory_def)
{
  SbsLayerPartition * partition = NULL;

  partition = (SbsLayerPartition *) malloc (sizeof(SbsLayerPartition));

  ASSERT (partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;

    memset (partition, 0x00, sizeof(SbsLayerPartition));

    /* Instantiate state_matrix */
    state_matrix = Multivector_new (memory_def,
                                    &SbsSettings_.state_matrix_format,
                                    3,
                                    rows,
                                    columns,
                                    vector_size);

    ASSERT(state_matrix != NULL);
    ASSERT(state_matrix->dimensionality == 3);
    ASSERT(state_matrix->data != NULL);
    ASSERT(state_matrix->dimension_size[0] == rows);
    ASSERT(state_matrix->dimension_size[1] == columns);
    ASSERT(state_matrix->dimension_size[2] == vector_size);

    partition->state_matrix = state_matrix;

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new (memory_def,
                                    &SbsSettings_.spike_matrix_format,
                                    2,
                                    rows,
                                    columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    partition->spike_matrix = spike_matrix;

    partition->x_pos = x_pos;
    partition->y_pos = y_pos;
  }

  return partition;
}

static void SbsLayerPartition_delete(SbsLayerPartition ** partition)
{
  ASSERT (partition != NULL);
  ASSERT (*partition != NULL);

  if ((partition != NULL) && (*partition != NULL))
  {
    Multivector_delete (&((*partition)->state_matrix));
    Multivector_delete (&((*partition)->spike_matrix));
    if ((*partition)->weight_matrix != NULL)
      Multivector_delete (&((*partition)->weight_matrix));

    free (*partition);

    *partition = NULL;
  }
}

static void SbsLayerPartition_initializeIP (SbsLayerPartition * partition,
                                            Multivector * state_matrix,
                                            NeuronState * state_vector,
                                            uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
	  float	initial_value_h = (1.0 / size);
    uint16_t 		neuron;
    for (neuron = 0; neuron < size; neuron ++)
    {
      switch (state_matrix->format.size)
      {
        case sizeof(uint32_t):
          ((uint32_t*) state_vector)[neuron] =
              (*(uint32_t *) (&initial_value_h)) >> (23 - state_matrix->format.mantissa_bitlength);
            break;
        case sizeof(uint16_t):
          ((uint16_t*) state_vector)[neuron] =
              (*(uint32_t *) (&initial_value_h)) >> (23 - state_matrix->format.mantissa_bitlength);
            break;
        default:
          ASSERT (0);
      }
    }
  }
}

static void SbsLayerPartition_initialize (SbsLayerPartition * partition,
                                          SbsLayerType layerType,
                                          uint32_t kernel_size)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = partition->state_matrix;
    uint16_t rows = state_matrix->dimension_size[0];
    uint16_t columns = state_matrix->dimension_size[1];
    uint16_t neurons = state_matrix->dimension_size[2];

    uint16_t row;
    uint16_t column;

    if (layerType != HX_INPUT_LAYER)
      for (row = 0; row < rows; row++)
        for (column = 0; column < columns; column++)
          SbsLayerPartition_initializeIP (partition,
                                          state_matrix,
                                          Multivector_2DAccess(state_matrix, row, column),
                                          neurons);
  }
}

static void SbsLayerPartition_cacheFlush (SbsLayerPartition * partition)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector_cacheFlush (partition->state_matrix);

    if (partition->weight_matrix != NULL)
      Multivector_cacheFlush (partition->weight_matrix);
  }
}

static void SbsLayerPartition_setWeights (SbsLayerPartition * partition,
                                          SbsWeightMatrix weight_matrix)
{
  ASSERT(partition != NULL);
  ASSERT(weight_matrix != NULL);

  if ((partition != NULL)
      && (weight_matrix != NULL))
  {
    if (partition->weight_matrix != NULL)
      Multivector_delete(&partition->weight_matrix);

    partition->weight_matrix =
        Multivector_duplicate(NULL,
                              weight_matrix);
  }
}

/*****************************************************************************/

static SbsLayer * SbsBaseLayer_new (SbsLayerType layer_type,
                                    uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    uint16_t kernel_stride,
                                    WeightShift weight_shift,
                                    MemoryBlock * memory_def)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);
  ASSERT(rows * columns <= 60 * 60);
  ASSERT(neurons <= 1024);

  if (layer != NULL)
  {
    int i;

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    layer->layer_type = layer_type;
    layer->num_partitions = 1;

    ASSERT (0 < layer->num_partitions);

    layer->partition_array = (SbsLayerPartition **)
        malloc (layer->num_partitions * sizeof(SbsLayerPartition *));
    ASSERT(layer->partition_array != NULL);

    if (layer->partition_array != NULL)
    {
      uint16_t residual = (rows % layer->num_partitions);
      uint16_t rows_per_partition = ((rows - residual) / layer->num_partitions);
      uint16_t rows_current_partition;
      uint16_t pos_y = 0;
      uint16_t pos_x = 0;

      ASSERT (((rows - residual) % layer->num_partitions) == 0);

      for (i = 0; i < layer->num_partitions; i++)
      {
        if (0 < residual)
        {
          rows_current_partition = rows_per_partition + 1;
          residual--;
        }
        else
        {
          rows_current_partition = rows_per_partition;
        }

        layer->partition_array[i] = SbsLayerPartition_new (pos_x,
                                                           pos_y,
                                                           rows_current_partition,
                                                           columns,
                                                           neurons,
                                                           memory_def);

        pos_y += rows_current_partition;
      }
    }

    if (1 < layer->num_partitions)
    { /* Instantiate spike_matrix */
      Multivector * spike_matrix = Multivector_new (NULL, &SbsSettings_.spike_matrix_format, 2, rows, columns);

      ASSERT(spike_matrix != NULL);
      ASSERT(spike_matrix->dimensionality == 2);
      ASSERT(spike_matrix->data != NULL);
      ASSERT(spike_matrix->dimension_size[0] == rows);
      ASSERT(spike_matrix->dimension_size[1] == columns);

      layer->spike_matrix = spike_matrix;
    }
    else
      layer->spike_matrix = layer->partition_array[0]->spike_matrix;

    /* Assign parameters */
    layer->rows          = rows;
    layer->columns       = columns;
    layer->vector_size   = neurons;
    layer->kernel_size   = kernel_size;
    layer->kernel_stride = kernel_stride;
    layer->weight_shift  = weight_shift;
  }

  return (SbsLayer *) layer;
}

static void SbsBaseLayer_delete(SbsLayer ** layer_ptr)
{
  ASSERT(layer_ptr!= NULL);
  ASSERT(*layer_ptr!= NULL);
  if ((layer_ptr != NULL) && (*layer_ptr != NULL))
  {
    SbsBaseLayer ** layer = (SbsBaseLayer **) layer_ptr;

    if ((*layer)->spike_matrix->memory_def_parent == NULL)
      Multivector_delete(&((*layer)->spike_matrix));

    if ((*layer)->partition_array != NULL)
      while ((*layer)->num_partitions)
        SbsLayerPartition_delete (&((*layer)->partition_array[--(*layer)->num_partitions]));

    free (*layer);
    *layer = NULL;
  }
}

static void SbsBaseLayer_initialize(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    int i;

    for (i = 0; i < layer->num_partitions; i++)
    {
      SbsLayerPartition_initialize (layer->partition_array[i],
                                    layer->layer_type,
                                    layer->kernel_size);


    }
  }
}

static void SbsBaseLayer_cacheFlush(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    int i;
    for (i = 0; i < layer->num_partitions; i++)
      SbsLayerPartition_cacheFlush(layer->partition_array[i]);
  }
}

static void SbsBaseLayer_giveWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if ((layer != NULL) && (((SbsBaseLayer *) layer)->partition_array != NULL))
  {
    int i;

    for (i = 0; i < ((SbsBaseLayer *) layer)->num_partitions; i++)
      SbsLayerPartition_setWeights (((SbsBaseLayer *) layer)->partition_array[i],
                                     weight_matrix);

    Multivector_delete ((Multivector**) &weight_matrix);
  }
}

static void SbsBaseLayer_setEpsilon(SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
  {
    SbsBaseLayer * base_layer = ((SbsBaseLayer *)layer);
    base_layer->epsilon = epsilon;
  }
}

static void SbsBaseLayer_setLearningRule (SbsLayer * layer_ptr, SbsLearningRule rule, double gama, int number_of_patterns)
{
  ASSERT (layer_ptr != NULL);
  ASSERT (0 < number_of_patterns);
  if ((layer_ptr != NULL) && (0 < number_of_patterns))
  {
    SbsBaseLayer * layer = (SbsBaseLayer *) layer_ptr;
    layer->learning_data.learning_rule = rule;
    layer->learning_data.number_of_patterns = number_of_patterns;
    layer->learning_data.current_pattern = 0;
    int w_spikes = layer->partition_array[0]->weight_matrix->dimension_size[2];

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.omega_matrix != NULL)
      Multivector_delete(&layer->learning_data.omega_matrix);

    layer->learning_data.omega_matrix = Multivector_new(NULL,
                                                        &SbsSettings_.learning_matrix_format,
                                                        2,
                                                        w_spikes,
                                                        layer->vector_size);

    ASSERT (layer->learning_data.omega_matrix != NULL);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.a_matrix != NULL)
      Multivector_delete(&layer->learning_data.a_matrix);

    layer->learning_data.a_matrix = Multivector_new(NULL,
                                                    &SbsSettings_.learning_matrix_format,
                                                    2,
                                                    w_spikes,
                                                    layer->vector_size);

    ASSERT (layer->learning_data.a_matrix != NULL);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.b_matrix != NULL)
      Multivector_delete(&layer->learning_data.b_matrix);

    layer->learning_data.b_matrix = Multivector_new(NULL,
                                                    &SbsSettings_.learning_matrix_format,
                                                    2,
                                                    w_spikes,
                                                    layer->vector_size);

    ASSERT (layer->learning_data.b_matrix != NULL);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.reco_vector != NULL)
      free(layer->learning_data.reco_vector);

    layer->learning_data.reco_vector = malloc(sizeof(float) * w_spikes);

    ASSERT (layer->learning_data.reco_vector != NULL);

    memset (layer->learning_data.reco_vector, 0, sizeof(float) * w_spikes);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.delat_vector != NULL)
      free(layer->learning_data.delat_vector);

    layer->learning_data.delat_vector = malloc(sizeof(float) * w_spikes);

    ASSERT (layer->learning_data.delat_vector != NULL);

    memset (layer->learning_data.delat_vector, 0, sizeof(float) * w_spikes);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.b_vector != NULL)
      free(layer->learning_data.b_vector);

    layer->learning_data.b_vector = malloc(sizeof(double) * layer->vector_size);

    ASSERT (layer->learning_data.b_vector != NULL);

    memset (layer->learning_data.b_vector, 0, sizeof(double) * layer->vector_size);

    ///////////////////////////////////////////////////////////////////////////

    layer->learning_data.gama = gama;
  }
}

inline static SpikeID SbsLayerPartition_stateVector_generateSpike (SbsLayerPartition * partition, NeuronState * state_vector, uint16_t size) __attribute__((always_inline));

inline static SpikeID SbsLayerPartition_stateVector_generateSpike (SbsLayerPartition * partition, NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    NeuronState random_s = (float)MT19937_genrand () / (float) 0xFFFFFFFF;
    NeuronState sum      = 0;
    SpikeID     spikeID;

    ASSERT(random_s <= 1.0);

    for (spikeID = 0; spikeID < size; spikeID++)
    {
      sum += state_vector[spikeID];

      //ASSERT(sum <= 1 + 1e-5);

      if (random_s <= sum)
        return spikeID;
    }
  }

  return size - 1;
}

static void SbsLayerPartition_loadInput(SbsLayerPartition * partition, char * file_name, uint8_t * input_label)
{
  ASSERT(partition != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  if ((partition != NULL) && (file_name != NULL) && (input_label != NULL))
  {
    FIL fil; /* File object */
    FRESULT rc;
    rc = f_open (&fil, file_name, FA_READ);
    ASSERT(rc == FR_OK);

    if (rc == FR_OK)
    {
      uint16_t  rows    = partition->state_matrix->dimension_size[0];
      uint16_t  columns = partition->state_matrix->dimension_size[1];
      uint16_t  neurons = partition->state_matrix->dimension_size[2];
      Multivector * fs_matrix = NULL;

      uint16_t row;
      uint16_t column;
      size_t   read_result = 0;

      uint8_t good_reading_flag = 1;

      if (memcmp(&partition->state_matrix->format,
                 &SbsSettings_.input_matrix_format_file_system,
                 sizeof(Format)) == 0)
      {
        fs_matrix = partition->state_matrix;
      }
      else
      {
        fs_matrix = Multivector_new(NULL, &SbsSettings_.input_matrix_format_file_system, 3, rows, columns, neurons);
      }

      size_t inference_population_size = fs_matrix->format.size * neurons;

      ASSERT (fs_matrix != NULL);

      for (column = 0; (column < columns) && good_reading_flag; column++)
        for (row = 0; (row < rows) && good_reading_flag; row++)
        {
          rc = f_read (&fil, Multivector_2DAccess(fs_matrix, row, column), inference_population_size, &read_result);

          good_reading_flag = read_result == inference_population_size;
        }

      if (good_reading_flag)
      {
        rc = f_read (&fil, input_label, sizeof(uint8_t), &read_result);
        (*input_label)--;
        good_reading_flag = read_result == sizeof(uint8_t);

        if (memcmp(&partition->state_matrix->format,
                   &fs_matrix->format,
                   sizeof(Format)) != 0)
        {

          Multivector * input_matrix_internal = Multivector_reformat (partition->state_matrix->memory_def_parent,
                                                                      fs_matrix,
                                                                      &partition->state_matrix->format);

          ASSERT(input_matrix_internal != NULL);

          Multivector_delete (&partition->state_matrix);

          partition->state_matrix = input_matrix_internal;
        }
      }

      f_close (&fil);
      ASSERT(good_reading_flag);
    }

  }
}

static void SbsBaseLayer_loadInput (SbsBaseLayer * layer, char * file_name,
                                    uint8_t * input_label)
{
  ASSERT(layer != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  ASSERT(layer->layer_type == HX_INPUT_LAYER);
  ASSERT(layer->num_partitions == 1);
  if ((layer != NULL) && (file_name != NULL) && (input_label != NULL))
  {
    SbsLayerPartition_loadInput (layer->partition_array[0], file_name,
                                 input_label);
  }
}

#define DATA16_TO_FLOAT32(d)   (0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12))

static void SbsBaseLayer_getOutputVector(SbsBaseLayer * layer,
                                         float *  output_vector,
                                         uint16_t output_vector_size)
{
  ASSERT(layer != NULL);
  ASSERT(0 < layer->num_partitions);
  ASSERT(layer->layer_type == HY_OUTPUT_LAYER);
  ASSERT(layer->partition_array[layer->num_partitions - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != 0);

  if ((layer != NULL) && (0 < layer->num_partitions)
      && (layer->layer_type == HY_OUTPUT_LAYER)
      && (layer->partition_array[layer->num_partitions - 1] != NULL)
      && (output_vector != NULL)
      && (output_vector_size != 0))
  {
    SbsLayerPartition *  partition = layer->partition_array[layer->num_partitions - 1];
    Multivector * output_state_matrix = partition->state_matrix;

    ASSERT(output_state_matrix->data != NULL);
    ASSERT(output_state_matrix->dimensionality == 3);
    ASSERT(output_state_matrix->dimension_size[0] == 1);
    ASSERT(output_state_matrix->dimension_size[1] == 1);
    ASSERT(0 < output_state_matrix->dimension_size[2]);

    for (int i = 0;
        (i < output_vector_size) && (i < output_state_matrix->dimension_size[2]);
        i++)
      switch (output_state_matrix->format.size)
      {
        case sizeof(uint16_t):
          ((uint32_t*) output_vector)[i] = DATA16_TO_FLOAT32(((uint16_t* )output_state_matrix->data)[i]);
          break;
        case sizeof(uint32_t):
          ((uint32_t*) output_vector)[i] = ((uint32_t*) output_state_matrix->data)[i];
          break;
        default:
          ASSERT(0);
      }
  }
}

inline SbsLayerPartition * SbsBaseLayer_getPartition(SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                              uint16_t * partition_row, uint16_t * partition_column) __attribute__((always_inline));

inline SbsLayerPartition * SbsBaseLayer_getPartition(SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                              uint16_t * partition_row, uint16_t * partition_column)
{
  SbsLayerPartition * partition = NULL;
  if (layer->num_partitions == 1)
  {
    partition = layer->partition_array[0];
    if (partition_row) *partition_row = row;
    if (partition_column) *partition_column = column;
  }
  else
  {
    int i;
    for (i = 0; partition == NULL && i < (layer)->num_partitions; i++)
    {
      if (layer->partition_array[i]->x_pos <= column
          && column < layer->partition_array[i]->x_pos + layer->partition_array[i]->state_matrix->dimension_size[1]
          && layer->partition_array[i]->y_pos <= row
          && row < layer->partition_array[i]->y_pos + layer->partition_array[i]->state_matrix->dimension_size[0])
      {
        partition = layer->partition_array[i];
        if (partition_row) *partition_row = row - layer->partition_array[i]->y_pos;
        if (partition_column) *partition_column = column - layer->partition_array[i]->x_pos;
      }
    }
  }

  return partition;
}

//#define ACCELERATOR

#ifdef ACCELERATOR

#include "xsbs_spike_master.h"

XSbs_spike_master InstancePtr;

void spike_master_InterruptHandler (void *data)
{
  uint32_t status;
  XSbs_spike_master * spike_master_instance = (XSbs_spike_master *) data;
  status = XSbs_spike_master_InterruptGetStatus (spike_master_instance);
  XSbs_spike_master_InterruptClear (spike_master_instance, status);
}

void spike_master_hw_initialize (SbsBaseLayer * layer)
{
  static uint8_t initialized_ = 0;

  if (!initialized_)
  {
    SbsLayerPartition * partition     = layer->partition_array[0];
    Multivector *       state_matrix  = partition->state_matrix;
    uint16_t            columns       = layer->columns;
    uint16_t            rows          = layer->rows;

    XSbs_spike_master_Initialize (&InstancePtr, XPAR_SBS_SPIKE_MASTER_0_DEVICE_ID);
    XSbs_spike_master_InterruptGlobalEnable (&InstancePtr);
    XSbs_spike_master_InterruptEnable (&InstancePtr, 1);
    ARM_GIC_connect (XPAR_FABRIC_SBS_SPIKE_MASTER_0_INTERRUPT_INTR, spike_master_InterruptHandler, &InstancePtr);

    XSbs_spike_master_Set_spike_matrix_mem (&InstancePtr, (unsigned int) layer->spike_matrix->data);
    XSbs_spike_master_Set_state_matrix_mem (&InstancePtr, (unsigned int) state_matrix->data);
    XSbs_spike_master_Set_rows (&InstancePtr, rows);
    XSbs_spike_master_Set_columns (&InstancePtr, columns);
    XSbs_spike_master_Set_vector_size (&InstancePtr, layer->vector_size);
    XSbs_spike_master_Set_seed (&InstancePtr, 666);

    initialized_ = 1;
  }
}
#else
#include "sbs_spike_master.h"
#endif

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0xFFFF & (((unsigned int)(d)) >> 12))

typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

static unsigned short SbsBaseLayer_generateSpikeIP(unsigned short * state_vector_mem, uint16_t size)
{
  float state_vector[1024] = { 0 };
  float random_s = ((float) MT19937_genrand ()) / ((float) 0xFFFFFFFF);
  float sum = 0.0f;
  unsigned short spikeID;

  for (int i = 0; i < size; i++)
  {
    *((unsigned int *) &state_vector[i]) = DATA16_TO_FLOAT32(state_vector_mem[i]);
  }

  for (spikeID = 0; spikeID < size; spikeID++)
  {
    sum += state_vector[spikeID];

    if (random_s <= sum)
      return spikeID;
  }

  return size - 1;
}

/////////////////////////////////////////////////////////////////////////////

#define FP_OR_MASK  (0x30000000)
#define FP_AND_MASK (0x0ffff000) | FP_OR_MASK


#define OR_MASK(reg)  {*((uint32_t*)&reg) |= FP_OR_MASK;}
#define AND_MASK(reg) {*((uint32_t*)&reg) &= FP_AND_MASK;}

#define FORCE(reg) {AND_MASK(reg); if ((*((uint32_t*)&reg) & FP_OR_MASK) != FP_OR_MASK) *((uint32_t*)&reg) = 0;}

static void SbsBaseLayer_updateIP_(float * state_vector_ptr, float * weight_vector_ptr, unsigned short size, float epsilon)
{
  float state_vector[1024];
  float weight_vector[1024];

  for (int i = 0; i < size; i ++)
  {
    state_vector[i] = state_vector_ptr[i];
    weight_vector[i] = weight_vector_ptr[i];
  }

  for (int i = 0; i < size; i ++)
  {
    if (*((unsigned int*) &weight_vector[i]) != 0)
    {
      if ((*((unsigned int*) &weight_vector[i]) & 0x38000000) != 0x38000000)
      {
        *((unsigned int*) &weight_vector[i]) = 0;
      }
      else if (~0x3ff80000 & *((unsigned int*) &weight_vector[i]))
      {
        *((unsigned int*) &weight_vector[i]) &= 0x3ff80000;
      }
    }

    if (*((unsigned int*) &weight_vector[i]) == 0)
    {
      *((unsigned int*) &weight_vector[i]) = 0x38000000;
    }
    FORCE(state_vector[i]);
  }

  FORCE(epsilon);

  if ((state_vector != NULL) && (weight_vector != NULL) && (0 < size))
  {
    float temp_data[1024]     = {0};

    float sum             = 0.0f;
    float reverse_epsilon = 1.0f / (1.0f + epsilon);
    float epsion_over_sum = 0.0f;
    unsigned short    neuron;

    FORCE(reverse_epsilon);

    float h;
    float p;
    float h_p;
    float h_new;

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

  }

  for (int i = 0; i < size; i ++)
  {
    state_vector_ptr[i] = state_vector[i];
    weight_vector_ptr[i] = weight_vector[i];
  }
}
////////////////////////////////////////////////////////////////////////////

static void SbsBaseLayer_updateIP(unsigned short * state_vector_mem, unsigned char * weight_vector_mem, unsigned int size, float epsilon)
{
  float state_vector[1024];
  float weight_vector[1024];

  for (int i = 0; i < size; i ++)
  {
    *((unsigned int *) &state_vector[i]) = DATA16_TO_FLOAT32(state_vector_mem[i]);
    *((unsigned int *) &weight_vector[i]) = DATA8_TO_FLOAT32(weight_vector_mem[i]);
  }

  SbsBaseLayer_updateIP_(state_vector, weight_vector, size, epsilon);
  for (int neuron = 0; neuron < size; neuron ++)
  {
    if ((*((unsigned int *)&state_vector[neuron]) & 0x30000000) == 0x30000000)
    {
      state_vector_mem[neuron] = FLOAT32_TO_DATA16(*((unsigned int * )&state_vector[neuron]));
    }
    else
    {
      state_vector_mem[neuron] = 0;
    }
  }
  return;

  float temp_data[1024] = { 0 };

  float sum = 0.0f;
  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  float epsion_over_sum = 0.0f;
  uint16_t neuron;


  float h;
  float p;
  float h_p;
  float h_new;

  for (neuron = 0; neuron < size; neuron++)
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

  for (neuron = 0; neuron < size; neuron++)
  {
    h_p = temp_data[neuron];
    h = state_vector[neuron];

    h_new = reverse_epsilon * (h + h_p * epsion_over_sum);
    state_vector[neuron] = h_new;
  }

  for (neuron = 0; neuron < size; neuron ++)
  {
    if ((*((unsigned int *)&state_vector[neuron]) & 0x30000000) == 0x30000000)
    {
      state_vector_mem[neuron] = FLOAT32_TO_DATA16(*((unsigned int * )&state_vector[neuron]));
    }
    else
    {
      state_vector_mem[neuron] = 0;
    }
  }
}


void sbs_spike_master_f(unsigned int * spike_matrix_mem,
                      unsigned int * state_matrix_mem,
                      unsigned int rows,
                      unsigned int columns,
                      unsigned int vector_size,
                      unsigned int seed,
                      unsigned int * debug_mem)
{
#pragma HLS INTERFACE m_axi port=spike_matrix_mem offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=state_matrix_mem offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=debug_mem        offset=slave bundle=gmem_debug

#pragma HLS INTERFACE s_axilite port=spike_matrix_mem bundle=control
#pragma HLS INTERFACE s_axilite port=state_matrix_mem bundle=control
#pragma HLS INTERFACE s_axilite port=debug_mem        bundle=control
#pragma HLS INTERFACE s_axilite port=rows             bundle=control
#pragma HLS INTERFACE s_axilite port=columns          bundle=control
#pragma HLS INTERFACE s_axilite port=vector_size      bundle=control
#pragma HLS INTERFACE s_axilite port=seed             bundle=control
#pragma HLS INTERFACE s_axilite port=return           bundle=control

  unsigned int   row;
  unsigned int   column;
  unsigned int   spike_matrix_offset;

  unsigned int   ip_index;
  float          random_s;
  float          sum;
  Data32         h;
  unsigned short spikeID;

  unsigned int  ip_vector[50];
#pragma HLS ARRAY_PARTITION variable=ip_vector complete dim=1
  unsigned int  spike_vector[32 * 32] = { 0 };
  unsigned char spike_vector_index;
  float         random;

  if (!MT19937_initialized ())
  {
    MT19937_sgenrand (seed);
  }

  for (row = 0; row < rows; row++)
  {
    for (column = 0; column < columns; column++)
    {
      spike_matrix_offset = row * columns + column;
      ip_index  = spike_matrix_offset * vector_size;

      memcpy (ip_vector, &state_matrix_mem[ip_index], sizeof(unsigned short) * vector_size);

      random    = MT19937_genrand ();
      random_s  = random / ((float) 0xFFFFFFFF);

      sum       = 0.0f;
      for (spikeID = 0;
           spikeID < vector_size;
           spikeID++)
      {
#pragma HLS PIPELINE
        if (sum < random_s)
        {
          h.u32 = DATA16_TO_FLOAT32(ip_vector[spikeID >> 1] >> ((spikeID & 1) * 16));

          sum += h.f32;

          if (random_s <= sum || (spikeID == vector_size - 1))
          {
            spike_vector[spike_matrix_offset >> 1] |= spikeID << ((spike_matrix_offset & 1) * 16);
          }
        }
      }
    }
  }

  memcpy (spike_matrix_mem, spike_vector, rows * columns * sizeof(unsigned short));
}


static void SbsBaseLayer_generateSpikes (SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(1 == layer->num_partitions);
  ASSERT(layer->layer_type == HX_INPUT_LAYER);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (1 == layer->num_partitions))
  {
    ASSERT (layer->partition_array[0] != NULL);
    ASSERT (layer->partition_array[0]->state_matrix != NULL);
    ASSERT (layer->partition_array[0]->spike_matrix != NULL);
    ASSERT (layer->rows == layer->partition_array[0]->state_matrix->dimension_size[0]);
    ASSERT (layer->columns == layer->partition_array[0]->state_matrix->dimension_size[1]);

#ifdef ACCELERATOR
    spike_master_hw_initialize (layer);

    Multivector_cacheFlush (layer->partition_array[0]->state_matrix);

    XSbs_spike_master_Start (&InstancePtr);

    while (!XSbs_spike_master_IsDone (&InstancePtr));

    Multivector_cacheInvalidate (layer->spike_matrix);
#else
    sbs_spike_master_f (layer->spike_matrix->data,
                      layer->partition_array[0]->state_matrix->data,
                      layer->rows,
                      layer->columns,
                      layer->vector_size,
                      666,
                      NULL);
#endif
  }
}


inline static void SbsBaseLayer_update(SbsBaseLayer * layer, SbsBaseLayer * spike_layer) __attribute__((always_inline));
inline static void SbsBaseLayer_update(SbsBaseLayer * layer, SbsBaseLayer * spike_layer)
{
  ASSERT (layer != NULL);
  ASSERT (spike_layer != NULL);
  if ((layer != NULL) && (spike_layer != NULL))
  {
    int i;
    SbsLayerPartition *  update_partition = NULL;
    uint16_t             update_partition_rows;
    uint16_t             update_partition_row;
    Multivector * spike_layer_spike_matrix = spike_layer->spike_matrix;

    unsigned short spikeID  = 0;
    unsigned int * weight_vector = NULL;
    unsigned int * state_vector = NULL;

    uint16_t kernel_stride  = layer->kernel_stride;
    uint16_t kernel_size    = layer->kernel_size;


    uint16_t layer_row;         /* Row index for navigation on the layer */
    uint16_t layer_column;      /* Column index for navigation on the layer */
    uint16_t kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
    uint16_t kernel_row_pos;    /* Kernel row position for navigation on the spike matrix */
    uint16_t kernel_row;        /* Row index for navigation inside kernel */
    uint16_t kernel_column;     /* Column index for navigation inside kernel */

    uint16_t layer_columns = layer->columns;

    Multivector * update_partition_weight_matrix = NULL;
    Multivector * update_partition_state_matrix = NULL;

    WeightShift layer_weight_shift = layer->weight_shift;

    kernel_row_pos = 0, layer_row = 0;
    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      update_partition_weight_matrix = update_partition->weight_matrix;
      update_partition_state_matrix = update_partition->state_matrix;
      update_partition_rows = update_partition_state_matrix->dimension_size[0];

      /* Update begins */
      for (update_partition_row = 0;
           update_partition_row < update_partition_rows;
           update_partition_row ++,
           kernel_row_pos += kernel_stride, layer_row ++)
      {
        for (kernel_column_pos = 0, layer_column = 0;
            layer_column < layer_columns;
             kernel_column_pos += kernel_stride, layer_column ++)
        {
          state_vector = Multivector_2DAccess(update_partition_state_matrix, update_partition_row, layer_column);

          if (layer->layer_type != HY_OUTPUT_LAYER)
            *(unsigned short *) Multivector_2DAccess(layer->spike_matrix, update_partition_row, layer_column) = SbsBaseLayer_generateSpikeIP(state_vector, layer->vector_size);

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              spikeID = *(unsigned short *) Multivector_2DAccess(spike_layer_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

              ASSERT(layer->vector_size == update_partition->weight_matrix->dimension_size[3]);

              if (layer_weight_shift == COLUMN_SHIFT)
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_row, kernel_column, spikeID);
              }
              else
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_column, kernel_row, spikeID);
              }

              SbsBaseLayer_updateIP (state_vector, weight_vector, layer->vector_size, layer->epsilon);
            }
          }
        }
      }
      /* Update ends */
    }
  }
}

/*****************************************************************************/

static SbsNetwork * SbsBaseNetwork_new(void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc (sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
    memset (network, 0x0, sizeof(SbsBaseNetwork));
    network->vtbl = _SbsNetwork;
    network->input_label = (uint8_t) -1;
    network->inferred_output = (uint8_t) -1;
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

    layer_array = realloc (layer_array, (size + 1) * sizeof(SbsBaseLayer *));

    ASSERT(layer_array != NULL);

    if (layer_array != NULL)
    {
      layer_array[size] = (SbsBaseLayer *) layer;

      network->layer_array = layer_array;
      network->size++;
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
      && (network->layer_array != NULL)
      && (*network->layer_array != NULL)
      && (file_name != NULL))
  {
    SbsBaseLayer_loadInput (network->layer_array[0],
                            file_name,
                            &network->input_label);
  }
}

#define SBS_LEARNING_THRESHOLD 0.00001

static void SbsBaseLayer_learningDeltaMSE(SbsBaseLayer * layer, SbsBaseLayer * prev_layer)
{
  return;// TODO: Update algorithm for custom floating point

  ASSERT (layer != NULL);
  ASSERT (layer->partition_array != NULL);
  ASSERT (layer->partition_array[0] != NULL);
  ASSERT (layer->partition_array[0]->state_matrix != NULL);
  ASSERT (layer->partition_array[0]->state_matrix->dimensionality == 3);
  ASSERT (layer->partition_array[0]->weight_matrix != NULL);
  ASSERT (layer->partition_array[0]->weight_matrix->dimensionality == 4);


  ASSERT (prev_layer != NULL);
  ASSERT (prev_layer->partition_array != NULL);
  ASSERT (prev_layer->partition_array[0] != NULL);
  ASSERT (prev_layer->partition_array[0]->state_matrix != NULL);
  ASSERT (prev_layer->partition_array[0]->state_matrix->dimensionality == 3);

  if (layer != NULL && prev_layer != NULL)
  {
    Multivector * h_matrix = layer->partition_array[0]->state_matrix;
    int h_neurons = h_matrix->dimension_size[2];

    Multivector * w_matrix = layer->partition_array[0]->weight_matrix;
    int w_rows = w_matrix->dimension_size[0];
    int w_cols = w_matrix->dimension_size[1];
    int w_spikes = w_matrix->dimension_size[2];
    int w_neurons = w_matrix->dimension_size[3];

    Multivector * prev_layer_h_matrix = prev_layer->partition_array[0]->state_matrix;

    int row;
    int col;
    int spike;
    int i;

    Weight w;
    NeuronState h;

    float * reco_vector = layer->learning_data.reco_vector;
    float * delat_vector = layer->learning_data.delat_vector;

    ASSERT (reco_vector != NULL);
    ASSERT (delat_vector != NULL);

    ASSERT (w_neurons == h_neurons);

    // 1)
    for (row = 0; row < w_rows; row++)
    {
      for (col = 0; col < w_cols; col++)
      {
        for (spike = 0; spike < w_spikes; spike++)
        {
          reco_vector[spike] = 0;
          for (i = 0; i < w_neurons; i++)
          {
            w = ((Weight *) Multivector_3DAccess (w_matrix, row, col, spike))[i];
            h = *((NeuronState *) Multivector_3DAccess (h_matrix, 0, 0, i));

            reco_vector[spike] += w * h;
          }
        }
      }
    }

    // 2)
    for (spike = 0; spike < w_spikes; spike++)
    {
      delat_vector[spike] = *(NeuronState*) Multivector_3DAccess (prev_layer_h_matrix, 0, 0, spike) - reco_vector[spike];
    }

    // 3)
    for (spike = 0; spike < w_spikes; spike++)
    {
      for (i = 0; i < h_neurons; i++)
      {
        h = *((NeuronState *) Multivector_3DAccess (h_matrix, 0, 0, i));

        *((double*) Multivector_2DAccess (layer->learning_data.omega_matrix, spike, i)) += ((double)delat_vector[spike]) * ((double)h);
      }
    }

    layer->learning_data.current_pattern ++;

    if (layer->learning_data.current_pattern == layer->learning_data.number_of_patterns)
    {
      double * b_vector = layer->learning_data.b_vector;
      double temp;
      ASSERT (b_vector != NULL);

      layer->learning_data.current_pattern = 0;

      temp = layer->learning_data.gama / layer->learning_data.number_of_patterns;

      for (spike = 0; spike < w_spikes; spike++)
      {
        for (i = 0; i < h_neurons; i++)
        {
          *((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)) =
              ((Weight*) Multivector_3DAccess (w_matrix, 0, 0, spike))[i] +
              temp * (*((double*) Multivector_2DAccess (layer->learning_data.omega_matrix, spike, i)));

          if (*((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)) < 0.0)
          {
            *((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)) = SBS_LEARNING_THRESHOLD;
          }
        }
      }

      memset (b_vector, 0, sizeof(double) * w_neurons);
      for (i = 0; i < w_neurons; i++)
      {
        for (spike = 0; spike < w_spikes; spike++)
        {
          b_vector[i] += *((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i));
        }
      }

      for (i = 0; i < w_neurons; i++)
      {
        temp = 1.0 / b_vector[i];
        for (spike = 0; spike < w_spikes; spike++)
        {
          ((Weight *) Multivector_3DAccess (w_matrix, 0, 0, spike))[i] =
              temp * (*((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)));

          ASSERT(0.0 <= ((Weight * ) Multivector_3DAccess (w_matrix, 0, 0, spike))[i]
                  && ((Weight * ) Multivector_3DAccess (w_matrix, 0, 0, spike))[i] <= 1.0);
        }
      }
    }
  }
}

static void SbsBaseLayer_learning(SbsBaseLayer * layer, SbsBaseLayer * prev_layer)
{
  ASSERT(layer != NULL);
  if (layer != NULL)
    switch (layer->learning_data.learning_rule)
    {
      case SBS_LEARNING_NONE:
        break;
      case SBS_LEARNING_DELTA_MSE:
        // TODO: MSE in fixed-point
        SbsBaseLayer_learningDeltaMSE(layer, prev_layer);
        break;
      case SBS_LEARNING_RELATIVE_ENTROPY:
        break;
      default:
        ASSERT (NULL);
    }
}

static void SbsBaseNetwork_updateInferredOutput(SbsBaseNetwork * network)
{
  ASSERT (network != NULL);

  if (network != NULL)
  {
    float max_value = 0;
    SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
    float output_state_vector[10] = {0};
    uint16_t output_vector_size = 10;
    float h;
    int i;

    SbsBaseLayer_getOutputVector (output_layer,
                                  output_state_vector,
                                  output_vector_size);

    ASSERT (output_state_vector != NULL);
    ASSERT (0 < output_vector_size);

    for (i = 0; i < output_vector_size; i++)
    {
      h = output_state_vector[i];

      if (max_value < h)
      {
        network->inferred_output = i;
        max_value = h;
      }
    }
  }
}

static void SbsBaseNetwork_updateCycle(SbsNetwork * network_ptr, uint16_t cycles)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  SbsBaseLayer * input_layer = NULL;

  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

#ifdef ACCELERATOR
  ARM_GIC_initialize ();
#endif

  if ((network != NULL)
      && (3 <= network->size)
      && (network->layer_array != NULL)
      && (cycles != 0))
  {
    int i;

    /* Initialize all layers except the input-layer */
    for (i = 0; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
      SbsBaseLayer_cacheFlush(network->layer_array[i]);
    }

    input_layer = network->layer_array[0];

    MT19937_sgenrand(666);

    /************************ Begins Update cycle ****************************/
    while (cycles--)
    {
      SbsBaseLayer_generateSpikes (input_layer);

      for (i = 1; i <= network->size - 1; i++)
      {
        SbsBaseLayer_update (network->layer_array[i],
                             network->layer_array[i - 1]);
      }
    }
    /************************ Ends Update cycle ******************************/

    /************************ Begins Learning cycle **************************/
    for (i = 1; i <= network->size - 1; i++)
    {
      SbsBaseLayer_learning (network->layer_array[i],
                             network->layer_array[i - 1]);
    }
    /************************ Ends Learning cycle ****************************/

    /************************ Ends Update cycle ****************************/

    /************************ Get inferred output **************************/
    SbsBaseNetwork_updateInferredOutput (network);
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

static void SbsBaseNetwork_getOutputVector(SbsNetwork * network_ptr,
                                           float * output_vector,
                                           uint16_t output_vector_size)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(0 < network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(network->layer_array[network->size - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != 0);

  if ((network != NULL)
      && (0 < network->size)
      && (network->layer_array != NULL)
      && (network->layer_array[network->size - 1] != NULL)
      && (output_vector != NULL)
      && (output_vector_size != 0))
  {
    SbsBaseLayer_getOutputVector (network->layer_array[network->size - 1],
                                  output_vector,
                                  output_vector_size);
  }
}

static void SbsBaseNetwork_printStatistics (SbsNetwork * network)
{

}
/*****************************************************************************/

MemoryBlock MemoryBlock_DDR =
  {
    .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x38000000,
    .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3BFFFFFF,
    .blockIndex  = 0
  };

static SbsLayer * SbsInputLayer_new(SbsLayerType layer_type,
                                    uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        0,
                                        0,
                                        ROW_SHIFT,
                                        &MemoryBlock_DDR);
}

static SbsLayer * SbsConvolutionLayer_new(SbsLayerType layer_type,
                                          uint16_t rows,
                                          uint16_t columns,
                                          uint16_t neurons,
                                          uint16_t kernel_size,
                                          WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        kernel_size,
                                        1,
                                        weight_shift,
                                        &MemoryBlock_DDR);
}

static SbsLayer * SbsPoolingLayer_new(SbsLayerType layer_type,
                                      uint16_t rows,
                                      uint16_t columns,
                                      uint16_t neurons,
                                      uint16_t kernel_size,
                                      WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        kernel_size,
                                        kernel_size,
                                        weight_shift,
                                        &MemoryBlock_DDR);
}

static SbsLayer * SbsFullyConnectedLayer_new(SbsLayerType layer_type,
                                             uint16_t neurons,
                                             uint16_t kernel_size,
                                             WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        1,
                                        1,
                                        neurons,
                                        kernel_size,
                                        1,
                                        weight_shift,
                                        &MemoryBlock_DDR);
}

static SbsLayer * SbsOutputLayer_new(SbsLayerType layer_type,
                                     uint16_t neurons,
                                     WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        1,
                                        1,
                                        neurons,
                                        1,
                                        1,
                                        weight_shift,
                                        &MemoryBlock_DDR);
}
/*****************************************************************************/

static SbsWeightMatrix SbsWeightMatrix_new (uint16_t rows,
                                            uint16_t columns,
                                            uint16_t depth,
                                            uint16_t neurons,
                                            char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new (&MemoryBlock_DDR,
                                     &SbsSettings_.weight_matrix_format_file_system,
                                     4,
                                     rows,
                                     columns,
                                     depth,
                                     neurons);

    ASSERT(weight_watrix != NULL);
    ASSERT(weight_watrix->dimensionality == 4);
    ASSERT(weight_watrix->data != NULL);
    ASSERT(weight_watrix->dimension_size[0] == rows);
    ASSERT(weight_watrix->dimension_size[1] == columns);
    ASSERT(weight_watrix->dimension_size[2] == depth);
    ASSERT(weight_watrix->dimension_size[3] == neurons);

    if ((weight_watrix != NULL)
        && (weight_watrix->dimensionality == 4)
        && (weight_watrix->data != NULL)
        && (weight_watrix->dimension_size[0] == rows)
        && (weight_watrix->dimension_size[1] == columns)
        && (weight_watrix->dimension_size[2] == depth)
        && (weight_watrix->dimension_size[3] == neurons))
    {
      FIL fil; /* File object */
      FRESULT rc;
      rc = f_open (&fil, file_name, FA_READ);
      ASSERT(rc == FR_OK);

      if (rc == FR_OK)
      {
        Multivector * weight_watrix_internal = NULL;
        size_t read_size;
        size_t data_size = rows * columns * depth * neurons * SbsSettings_.weight_matrix_format_file_system.size; // TODO: Define the data type of the file
        rc = f_read (&fil, weight_watrix->data, data_size, &read_size);
        ASSERT((rc == FR_OK) && (read_size == data_size));
        f_close (&fil);

        if (memcmp(&weight_watrix->format,
                   &SbsSettings_.weight_matrix_format,
                   sizeof(SbsSettings_.weight_matrix_format)) != 0)
        {
          weight_watrix_internal = Multivector_reformat (weight_watrix->memory_def_parent,
                                                         weight_watrix,
                                                         &SbsSettings_.weight_matrix_format);

          ASSERT(weight_watrix_internal != NULL);

          Multivector_delete (&weight_watrix);

          weight_watrix = weight_watrix_internal;
        }
      }
      else Multivector_delete (&weight_watrix);
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
                          SbsBaseNetwork_printStatistics};

SbsLayer _SbsLayer = {SbsBaseLayer_new,
                      SbsBaseLayer_delete,
                      SbsBaseLayer_setEpsilon,
                      SbsBaseLayer_setLearningRule,
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
