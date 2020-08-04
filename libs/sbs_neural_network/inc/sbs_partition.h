/*
 * sbs_partition.h
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */

#ifndef SBS_PARTITION_H_
#define SBS_PARTITION_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <result.h>

#include "sbs_processing_unit.h"
#include "sbs_weight_matrix.h"

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */

typedef uint16_t SpikeID;

typedef struct
{
  SbSUpdateAccelerator *  accelerator;
  SbsAcceleratorProfie *  profile;
  SbsLayerType            layerType;
  Event *                 event;
  Event *                 hw_processing_event;
  uint16_t      x_pos;
  uint16_t      y_pos;
  Multivector * state_matrix;
  Multivector * spike_matrix;
  Multivector * weight_matrix;
} SbsLayerPartition;

#pragma pack(pop)   /* restore original alignment from stack */

SbsLayerPartition * SbsLayerPartition_new (SbSUpdateAccelerator * accelerator,
                                           SbsLayerType layerType,
                                           uint16_t x_pos,
                                           uint16_t y_pos,
                                           uint16_t rows,
                                           uint16_t columns,
                                           uint16_t vector_size,
                                           Event * parent_event);

void SbsLayerPartition_delete (SbsLayerPartition ** partition);

void SbsLayerPartition_initialize (SbsLayerPartition * partition,
                                   uint32_t kernel_size,
                                   float epsilon,
                                   MemoryCmd accelerator_memory_cmd);

void SbsLayerPartition_cacheFlush (SbsLayerPartition * partition);

void SbsLayerPartition_setWeights (SbsLayerPartition * partition,
                                   SbsWeightMatrix weight_matrix);

Result SbsLayerPartition_loadInput (SbsLayerPartition * partition,
                                    char * file_name,
                                    uint8_t * input_label);

SpikeID SbsLayerPartition_stateVector_generateSpikeSw (uint32_t * state_vector,
                                                       uint16_t size);


#ifdef __cplusplus
}
#endif
#endif /* SBS_PARTITION_H_ */
