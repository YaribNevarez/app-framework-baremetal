/*
 * sbs_partition.c
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */


//#define DEBUG //  -g3 -O0 -DDEBUG

#include "sbs_partition.h"
#include "sbs_settings.h"

#include "mt19937int.h"
#include "miscellaneous.h"
#include "custom_float.h"

#include "ff.h"


/*****************************************************************************/
/*****************************************************************************/

SbsLayerPartition * SbsLayerPartition_new (SbSUpdateAccelerator * accelerator,
                                            uint16_t x_pos,
                                            uint16_t y_pos,
                                            uint16_t rows,
                                            uint16_t columns,
                                            uint16_t vector_size,
                                            Event * parent_event)
{
  SbsLayerPartition * partition = NULL;

  partition = (SbsLayerPartition *) malloc (sizeof(SbsLayerPartition));

  ASSERT (partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;
    MemoryBlock * memory_def = NULL;
    size_t        channelSize = 4; // Padding size in bytes

    memset (partition, 0x00, sizeof(SbsLayerPartition));

    if (accelerator != NULL)
    {
      ASSERT (accelerator->hardwareConfig != NULL);

      if (accelerator->hardwareConfig != NULL)
      {
        partition->accelerator = accelerator;
        memory_def = &accelerator->hardwareConfig->ddrMem;
        channelSize = accelerator->hardwareConfig->channelSize;
      }
    }

    /* Instantiate state_matrix */
    state_matrix = Multivector_new (memory_def,
                                    &SbsSettings_.state_matrix_format,
                                    channelSize,
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
                                    channelSize,
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

    partition->event = Event_new (parent_event, "Software");
  }

  return partition;
}

void SbsLayerPartition_delete(SbsLayerPartition ** partition)
{
  ASSERT (partition != NULL);
  ASSERT (*partition != NULL);

  if ((partition != NULL) && (*partition != NULL))
  {
    Multivector_delete (&((*partition)->state_matrix));
    Multivector_delete (&((*partition)->spike_matrix));
    if ((*partition)->weight_matrix != NULL)
      Multivector_delete (&((*partition)->weight_matrix));

    if ((*partition)->profile != NULL)
      SbsAcceleratorProfie_delete (&(*partition)->profile);

    if ((*partition)->hw_processing_event != NULL)
      Event_delete (&(*partition)->hw_processing_event);

    if ((*partition)->event != NULL)
      Event_delete (&(*partition)->event);

    free (*partition);

    *partition = NULL;
  }
}

static void SbsLayerPartition_initializeIP (SbsLayerPartition * partition,
                                            Multivector * state_matrix,
                                            float * state_vector,
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

void SbsLayerPartition_initialize (SbsLayerPartition * partition,
                                   SbsLayerType layerType,
                                   uint32_t kernel_size,
                                   float epsilon,
                                   MemoryCmd accelerator_memory_cmd)
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

    if (partition->accelerator != NULL)
    {
      if (partition->profile == NULL)
      {
        partition->profile = SbsAcceleratorProfie_new (layerType,
                                                       state_matrix,
                                                       partition->weight_matrix,
                                                       partition->spike_matrix,
                                                       kernel_size,
                                                       epsilon,
                                                       accelerator_memory_cmd,
                                                       partition->event);
      }
    }
    else
    {
      if (partition->hw_processing_event == NULL)
      {
        partition->hw_processing_event = Event_new (partition->event, "Hardware");
      }
    }
  }
}

void SbsLayerPartition_cacheFlush (SbsLayerPartition * partition)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector_cacheFlush (partition->state_matrix);

    if (partition->weight_matrix != NULL)
      Multivector_cacheFlush (partition->weight_matrix);
  }
}

void SbsLayerPartition_setWeights (SbsLayerPartition * partition,
                                   SbsWeightMatrix weight_matrix)
{
  ASSERT(partition != NULL);
  ASSERT(weight_matrix != NULL);

  if ((partition != NULL) && (weight_matrix != NULL))
  {
    MemoryBlock * memory_def = NULL;

    if ((partition->accelerator != NULL)
        && (partition->accelerator->hardwareConfig != NULL))
    {
      memory_def = &partition->accelerator->hardwareConfig->ddrMem;
    }

    if (partition->weight_matrix != NULL)
    {
      Multivector_delete (&partition->weight_matrix);
    }

    partition->weight_matrix = Multivector_duplicate (memory_def,
                                                      weight_matrix);
  }
}

Result SbsLayerPartition_loadInput (SbsLayerPartition * partition,
                                    char * file_name,
                                    uint8_t * input_label)
{
  Result result = ERROR;
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
      static Multivector * fs_matrix = NULL;

      uint16_t row;
      uint16_t column;
      size_t   read_result = 0;

      uint8_t good_reading_flag = 1;

      if (fs_matrix == NULL)
      {
        size_t memory_padding = 4;

        if (partition->accelerator != NULL)
        {
          ASSERT (partition->accelerator->hardwareConfig != NULL);
          memory_padding = partition->accelerator->hardwareConfig->channelSize;
        }

        fs_matrix = Multivector_new (NULL,
                                     &SbsSettings_.input_matrix_format_file_system,
                                     memory_padding,
                                     3,
                                     rows,
                                     columns,
                                     neurons);
      }
      else
      {
        Multivector_clear (fs_matrix);
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

        result = Multivector_copy (partition->state_matrix, fs_matrix);

        ASSERT (result == OK);
      }

      f_close (&fil);
      ASSERT(good_reading_flag);
    }
  }

  return result;
}

SpikeID SbsLayerPartition_stateVector_generateSpikeSw (uint32_t * state_vector,
                                                       uint16_t size)
{
  ASSERT (state_vector != NULL);
  ASSERT (0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    uint16_t *state_vector_ptr = (uint16_t *) state_vector;
    float random_s = (float) MT19937_genrand () / (float) 0xFFFFFFFF;
    float temp;
    float sum = 0;
    uint16_t spikeID;

    ASSERT(random_s <= 1.0);

    for (spikeID = 0; spikeID < size; spikeID++)
    { // promote
      *(uint32_t*) (&temp) = DATA16_TO_FLOAT32(state_vector_ptr[spikeID]);
      sum += temp;

      if (random_s <= sum)
        return spikeID;
    }
  }

  return size - 1;
}
