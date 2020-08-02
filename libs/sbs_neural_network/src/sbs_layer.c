/*
 * sbs_layer.c
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */


//#define DEBUG //  -g3 -O0 -DDEBUG



#include "event.h"
#include "miscellaneous.h"
#include "custom_float.h"

#include "sbs_layer.h"
#include "sbs_settings.h"


static SbsLayer _SbsLayer = { SbsBaseLayer_new,
                              SbsBaseLayer_delete,
                              SbsBaseLayer_setEpsilon,
                              SbsBaseLayer_giveWeights };

/*****************************************************************************/
static void SbsBaseLayer_updateHw (SbsBaseLayer * layer, SbsBaseLayer * spike_layer);
static void SbsBaseLayer_updateSw (SbsBaseLayer * layer, SbsBaseLayer * spike_layer);

static void SbsBaseLayer_generateSpikesHw (SbsBaseLayer * layer, SbsBaseLayer * dummy);
static void SbsBaseLayer_generateSpikesSw (SbsBaseLayer * layer, SbsBaseLayer * dummy);

SbsLayer * SbsBaseLayer_new(SbsLayerType layer_type,
                            uint16_t rows,
                            uint16_t columns,
                            uint16_t neurons,
                            uint16_t kernel_size,
                            uint16_t kernel_stride,
                            WeightShift weight_shift)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);
  ASSERT(rows * columns <= 60 * 60);
  ASSERT(neurons <= 1024);

  if (layer != NULL)
  {
    int i;
    SbSUpdateAccelerator * processor_group[/*NUM_ACCELERATOR_INSTANCES*/10] = { 0 };
    int processor_group_count = 0;

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    layer->event = Event_new (NULL, SbsLayerType_string (layer_type));

    layer->layer_type = layer_type;
    layer->num_partitions = 1;

    processor_group_count = SbSUpdateAccelerator_getGroupFromList ( layer_type, processor_group,
        sizeof(processor_group) / sizeof(SbSUpdateAccelerator *));

    if (0 < processor_group_count)
    {
      layer->num_partitions = processor_group_count;
      if (layer_type == HX_INPUT_LAYER)
      {
        layer->process = SbsBaseLayer_generateSpikesHw;
      }
      else
      {
        layer->process = SbsBaseLayer_updateHw;
      }
    }
    else
    {
      layer->num_partitions = 1;
      if (layer_type == HX_INPUT_LAYER)
      {
        layer->process = SbsBaseLayer_generateSpikesSw;
      }
      else
      {
        layer->process = SbsBaseLayer_updateSw;
      }
    }

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

      SbSUpdateAccelerator * accelerator = NULL;
      for (i = 0; i < layer->num_partitions; i++)
      {
        if (0 < processor_group_count)
        {
          accelerator = processor_group[i];
        }

        if (0 < residual)
        {
          rows_current_partition = rows_per_partition + 1;
          residual--;
        }
        else
        {
          rows_current_partition = rows_per_partition;
        }

        layer->partition_array[i] = SbsLayerPartition_new (accelerator,
                                                           pos_x,
                                                           pos_y,
                                                           rows_current_partition,
                                                           columns,
                                                           neurons,
                                                           layer->event);

        pos_y += rows_current_partition;
      }
    }

    if (1 < layer->num_partitions)
    { /* Instantiate spike_matrix */
      Multivector * spike_matrix = Multivector_new (NULL,
                                                    &SbsSettings_.spike_matrix_format,
                                                    layer->partition_array[0]->accelerator->hardwareConfig->channelSize,
                                                    2,
                                                    rows,
                                                    columns);

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

void SbsBaseLayer_delete (SbsLayer ** layer_ptr)
{
  ASSERT(layer_ptr!= NULL);
  ASSERT(*layer_ptr!= NULL);
  if ((layer_ptr != NULL) && (*layer_ptr != NULL))
  {
    SbsBaseLayer ** layer = (SbsBaseLayer **) layer_ptr;

    if ((*layer)->spike_matrix->memory_def_parent == NULL)
      Multivector_delete (&((*layer)->spike_matrix));

    if ((*layer)->partition_array != NULL)
      while ((*layer)->num_partitions)
        SbsLayerPartition_delete (&((*layer)->partition_array[--(*layer)->num_partitions]));

    if ((*layer)->event != NULL)
      Event_delete (&(*layer)->event);

    free (*layer);
    *layer = NULL;
  }
}

void SbsBaseLayer_initialize (SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    MemoryCmd accelerator_memory_cmd = { .cmdID = MEM_CMD_NONE };
    int i;

    if (1 < layer->num_partitions)
    {
      accelerator_memory_cmd.cmdID = MEM_CMD_MOVE;
      accelerator_memory_cmd.dest = layer->spike_matrix->data;
    }

    for (i = 0; i < layer->num_partitions; i++)
    {
      accelerator_memory_cmd.src = layer->partition_array[i]->spike_matrix->data;
      accelerator_memory_cmd.size = Multivector_dataSize(layer->partition_array[i]->spike_matrix);

      SbsLayerPartition_initialize (layer->partition_array[i],
                                    layer->layer_type,
                                    layer->kernel_size,
                                    layer->epsilon,
                                    accelerator_memory_cmd);

      accelerator_memory_cmd.dest += accelerator_memory_cmd.size;
    }
  }
}

void SbsBaseLayer_cacheFlush (SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL) && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    int i;
    for (i = 0; i < layer->num_partitions; i++)
      SbsLayerPartition_cacheFlush (layer->partition_array[i]);
  }
}

void SbsBaseLayer_giveWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
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

void SbsBaseLayer_setEpsilon (SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*//* 0.0 is allowed? */

  if (layer != NULL)
  {
    SbsBaseLayer * base_layer = ((SbsBaseLayer *) layer);
    base_layer->epsilon = epsilon;
  }
}

void SbsBaseLayer_loadInput (SbsBaseLayer * layer,
                             char * file_name,
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

void SbsBaseLayer_getOutputVector (SbsBaseLayer * layer,
                                   float * output_vector,
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

inline SbsLayerPartition * SbsBaseLayer_getPartition (SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                                      uint16_t * partition_row,
                                                      uint16_t * partition_column) __attribute__((always_inline));

inline SbsLayerPartition * SbsBaseLayer_getPartition (SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                                      uint16_t * partition_row,
                                                      uint16_t * partition_column)
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

static void SbsBaseLayer_generateSpikesHw (SbsBaseLayer * layer,
                                           SbsBaseLayer * dummy)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(1 == layer->num_partitions);
  ASSERT(layer->layer_type == HX_INPUT_LAYER);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (1 == layer->num_partitions))
  {
    SbsLayerPartition * partition     = layer->partition_array[0];
#ifdef DEBUG
    Multivector *       state_matrix  = partition->state_matrix;
    uint16_t            columns       = layer->columns;
    uint16_t            rows          = layer->rows;
#endif
    //uint16_t            row;
    //uint16_t            column;

    ASSERT (layer->partition_array[0] != NULL);
    ASSERT (layer->partition_array[0]->state_matrix != NULL);
    ASSERT (layer->partition_array[0]->spike_matrix != NULL);
    ASSERT (rows == state_matrix->dimension_size[0]);
    ASSERT (columns == state_matrix->dimension_size[1]);

    Event_start (layer->event);
    Event_start (layer->partition_array[0]->event);
    Accelerator_setup (partition->accelerator, partition->profile);
/*
    for (row = 0; row < rows; row++)
      for (column = 0; column < columns; column++)
        Accelerator_giveStateVector (partition->accelerator,
                                     Multivector_2DAccess (state_matrix, row, column));
*/
    Accelerator_start (partition->accelerator);
    Event_stop (layer->partition_array[0]->event);
    Event_stop (layer->event);
  }
}

void SbsBaseLayer_initializeHardware (SbsBaseLayer * layer)
{
  ASSERT (layer != NULL);
  if (layer != NULL)
  {
    int i;
    SbsLayerPartition *  update_partition = NULL;
    Multivector * update_partition_weight_matrix = NULL;

    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      if (update_partition->accelerator != NULL)
      {
        SbSUpdateAccelerator * update_partition_accelerator = update_partition->accelerator;
        switch (layer->layer_type)
        {
          case HX_INPUT_LAYER:
            break;
          case H1_CONVOLUTION_LAYER:
          case H3_CONVOLUTION_LAYER:
            update_partition_weight_matrix = update_partition->weight_matrix;
            ASSERT(update_partition_weight_matrix != NULL);

            Accelerator_loadCoefficients (update_partition_accelerator,
                                          update_partition->profile,
                                          update_partition_weight_matrix,
                                          layer->weight_shift == COLUMN_SHIFT);
            break;
          case H2_POOLING_LAYER:
          case H4_POOLING_LAYER:
            break;
          case H5_FULLY_CONNECTED_LAYER:
            break;
          case HY_OUTPUT_LAYER:
            break;
          default:;
        }
      }
    }
  }
}

static void SbsBaseLayer_generateSpikesSw (SbsBaseLayer * layer,
                                           SbsBaseLayer * dummy)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);
  ASSERT(layer->spike_matrix != NULL);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions)
      && (layer->spike_matrix != NULL))
  {
    int i;
    uint16_t columns = layer->columns;
    uint16_t neurons = layer->vector_size;

    uint16_t partition_row = 0;
    SbsLayerPartition * partition = NULL;
    Multivector * partition_state_matrix = NULL;
    Multivector* layer_spike_matrix = layer->spike_matrix;

    SpikeID * spike;
    uint32_t * state_vector;

    uint16_t row;
    uint16_t column;

    Event_start (layer->event);

    for (i = 0; i < layer->num_partitions; i++)
    {
      ASSERT(layer->partition_array[i] != NULL);
      ASSERT(layer->partition_array[i]->state_matrix != NULL);
      partition = layer->partition_array[i];
      partition_state_matrix = partition->state_matrix;
      Event_start (partition->event);
      for (row = partition->y_pos, partition_row = 0;
          partition_row < partition_state_matrix->dimension_size[0];
          partition_row++, row++)
      {
        for (column = 0; column < columns; column++)
        {
          spike = Multivector_2DAccess (layer_spike_matrix, row, column);
          state_vector = Multivector_2DAccess (partition_state_matrix,
                                               partition_row, column);
          *spike = SbsLayerPartition_stateVector_generateSpikeSw (state_vector,
                                                                  neurons);
        }
      }
      Event_stop (partition->event);
    }
    Event_stop (layer->event);
  }
}

static void SbsBaseLayer_updateIP (SbsBaseLayer * layer,
                                   uint32_t * state_vector,
                                   Weight * weight_vector,
                                   uint16_t size,
                                   float epsilon)
{
  ASSERT(state_vector != NULL);
  ASSERT(weight_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (weight_vector != NULL) && (0 < size))
  {
    uint16_t *state_vector_ptr = (uint16_t *) state_vector;
    uint8_t * weight_vector_ptr = (uint8_t *) weight_vector;

    static float temp_data[1024];

    float sum             = 0.0f;
    float reverse_epsilon = 1.0f / (1.0f + epsilon);
    float epsion_over_sum = 0.0f;
    uint16_t    neuron;

    /* Support for unaligned accesses in ARM architecture */
    float h;
    float p;
    float h_p;
    float h_new;

    for (neuron = 0; neuron < size; neuron ++)
    {
      *(uint32_t*) (&h) = DATA16_TO_FLOAT32(state_vector_ptr[neuron]);
      *(uint32_t*) (&p) = DATA08_TO_FLOAT32(weight_vector_ptr[neuron]);
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

      *(uint32_t*) (&h) = DATA16_TO_FLOAT32(state_vector_ptr[neuron]);

      h_new = reverse_epsilon * (h + h_p * epsion_over_sum);

      state_vector_ptr[neuron] = FLOAT32_TO_DATA16(*(uint32_t* )(&h_new));
    }
  }
}

static void SbsBaseLayer_updateSw (SbsBaseLayer * layer, SbsBaseLayer * spike_layer)
{
  ASSERT (layer != NULL);
  ASSERT (spike_layer != NULL);
  if ((layer != NULL) && (spike_layer != NULL))
  {
    int i;
    SbsLayerPartition * update_partition = NULL;
    uint16_t            update_partition_rows;
    uint16_t            update_partition_row;
    Multivector *       spike_layer_spike_matrix = spike_layer->spike_matrix;
    Multivector*        layer_spike_matrix = layer->spike_matrix;
    uint16_t            vector_size = layer->vector_size;
    float               epsilon = *((float*) &layer->epsilon);


    SpikeID     spikeID       = 0;
    Weight *    weight_vector = NULL;
    uint32_t *  state_vector  = NULL;
    SpikeID *   new_spike;


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

    Event_start (layer->event);
    //while (!spike_layer->partition_array[0]->accelerator->rxDone);

    kernel_row_pos = 0, layer_row = 0;
    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      Event_start (update_partition->event);
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
          new_spike = Multivector_2DAccess (layer_spike_matrix, update_partition_row, layer_column);
          state_vector = Multivector_2DAccess(update_partition_state_matrix, update_partition_row, layer_column);

          *new_spike = SbsLayerPartition_stateVector_generateSpikeSw (state_vector, vector_size);

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              spikeID = *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

              ASSERT (spikeID < spike_layer->vector_size);

              ASSERT(layer->vector_size == update_partition->weight_matrix->dimension_size[3]);

              if (layer_weight_shift == COLUMN_SHIFT)
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_row, kernel_column, spikeID);
              }
              else
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_column, kernel_row, spikeID);
              }

              SbsBaseLayer_updateIP (layer, state_vector, weight_vector, vector_size, epsilon);
            }
          }
        }
      }
      Event_stop (update_partition->event);
      /* Update ends */
    }
    Event_stop (layer->event);
  }
}

///////////////////////////////////////////////////////////////////////////////

static void SbsBaseLayer_updateHw (SbsBaseLayer * layer, SbsBaseLayer * spike_layer)
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

    SpikeID     spikeID       = 0;
    Weight *    weight_vector = NULL;
    uint32_t *  state_vector  = NULL;


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
    SbSUpdateAccelerator * update_partition_accelerator = NULL;
    Multivector * update_partition_state_matrix = NULL;

    WeightShift layer_weight_shift = layer->weight_shift;

    Event_start (layer->event);
    //while (!spike_layer->partition_array[0]->accelerator->rxDone);

    kernel_row_pos = 0, layer_row = 0;
    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      Event_start (update_partition->event);

      update_partition_weight_matrix = update_partition->weight_matrix;
      update_partition_accelerator = update_partition->accelerator;
      update_partition_state_matrix = update_partition->state_matrix;
      update_partition_rows = update_partition_state_matrix->dimension_size[0];

      Accelerator_setup (update_partition_accelerator,
                         update_partition->profile);

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

          Accelerator_giveStateVector (update_partition_accelerator, state_vector);

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              spikeID = *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

              ASSERT (spikeID < spike_layer->vector_size);

              if (layer->layer_type == H2_POOLING_LAYER
                  || layer->layer_type == H4_POOLING_LAYER
                  || layer->layer_type == H1_CONVOLUTION_LAYER
                  || layer->layer_type == H3_CONVOLUTION_LAYER)
              {
                Accelerator_giveSpike (update_partition_accelerator, spikeID);
              }
              else
              {
                ASSERT(layer->vector_size == update_partition->weight_matrix->dimension_size[3]);

                if (layer_weight_shift == COLUMN_SHIFT)
                {
                  weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_row, kernel_column, spikeID);
                }
                else
                {
                  weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_column, kernel_row, spikeID);
                }

                Accelerator_giveWeightVector (update_partition_accelerator, weight_vector);
              }
            }
          }
        }
      }
      /* Update ends */
      Accelerator_start (update_partition_accelerator);
      Event_stop (update_partition->event);
    }
    Event_stop (layer->event);
  }
}

void SbsBaseLayer_setParentEvent (SbsBaseLayer * layer,
                                  Event * parent_event)
{
  ASSERT (layer != NULL);
  if (layer != NULL)
  {
    Event_setParent (layer->event, parent_event);
  }
}


/*****************************************************************************/

SbsLayer * SbsInputLayer_new (SbsLayerType layer_type,
                              uint16_t rows,
                              uint16_t columns,
                              uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        0,
                                        0, ROW_SHIFT);
}

SbsLayer * SbsConvolutionLayer_new (SbsLayerType layer_type,
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
                                        weight_shift);
}

SbsLayer * SbsPoolingLayer_new (SbsLayerType layer_type,
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
                                        weight_shift);
}

SbsLayer * SbsFullyConnectedLayer_new (SbsLayerType layer_type,
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
                                        weight_shift);
}

SbsLayer * SbsOutputLayer_new (SbsLayerType layer_type,
                               uint16_t neurons,
                               WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        1,
                                        1,
                                        neurons,
                                        1,
                                        1,
                                        weight_shift);
}


