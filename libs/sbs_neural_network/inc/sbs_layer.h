/*
 * sbs_layer.h
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */

#ifndef SBS_LAYER_H_
#define SBS_LAYER_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <result.h>

#include "sbs_partition.h"
#include "sbs_weight_matrix.h"
#include "sbs_processing_unit.h"


typedef enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
} WeightShift;


#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */


typedef uint8_t   Weight;
typedef uint32_t  Random32;
typedef uint16_t  SpikeID;

typedef struct SbsLayer_ SbsLayer;
struct SbsLayer_
{
  SbsLayer * (*new)        (SbsLayerType layer_type,
                            uint16_t rows,
                            uint16_t columns,
                            uint16_t neurons,
                            uint16_t kernel_size,
                            uint16_t kernel_stride,
                            WeightShift weight_shift);
  void       (*delete)     (SbsLayer ** layer);
  void       (*setEpsilon) (SbsLayer * layer, float epsilon);
  void       (*giveWeights)(SbsLayer * layer, SbsWeightMatrix weight_matrix);
};


typedef struct _SbsBaseLayer SbsBaseLayer;

typedef void (*SbsProcessingFunction) (SbsBaseLayer * layer, SbsBaseLayer * spike_layer);

struct _SbsBaseLayer
{
  SbsLayer              vtbl;
  SbsProcessingFunction process;
  SbsLayerType          layer_type;
  SbsLayerPartition **  partition_array;
  Event *               event;
  uint8_t               num_partitions;
  uint16_t              rows;
  uint16_t              columns;
  uint16_t              vector_size;
  uint16_t              kernel_size;
  uint16_t              kernel_stride;
  WeightShift           weight_shift;
  float                 epsilon;
  Multivector *         spike_matrix;
};

#pragma pack(pop)   /* restore original alignment from stack */


SbsLayer * SbsBaseLayer_new (SbsLayerType layer_type,
                             uint16_t rows,
                             uint16_t columns,
                             uint16_t neurons,
                             uint16_t kernel_size,
                             uint16_t kernel_stride,
                             WeightShift weight_shift);

SbsLayer * SbsInputLayer_new (SbsLayerType layer_type,
                              uint16_t rows,
                              uint16_t columns,
                              uint16_t neurons);

SbsLayer * SbsConvolutionLayer_new (SbsLayerType layer_type,
                                    uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift);

SbsLayer * SbsPoolingLayer_new (SbsLayerType layer_type,
                                uint16_t rows,
                                uint16_t columns,
                                uint16_t neurons,
                                uint16_t kernel_size,
                                WeightShift weight_shift);

SbsLayer * SbsFullyConnectedLayer_new (SbsLayerType layer_type,
                                       uint16_t neurons,
                                       uint16_t kernel_size,
                                       WeightShift weight_shift);

SbsLayer * SbsOutputLayer_new (SbsLayerType layer_type,
                               uint16_t neurons,
                               WeightShift weight_shift);

void SbsBaseLayer_delete (SbsLayer ** layer_ptr);

void SbsBaseLayer_setEpsilon (SbsLayer * layer, float epsilon);

void SbsBaseLayer_giveWeights (SbsLayer * layer, SbsWeightMatrix weight_matrix);

void SbsBaseLayer_setParentEvent (SbsBaseLayer * layer,
                                  Event * parent_event);

Result SbsBaseLayer_loadInput (SbsBaseLayer * layer,
                               char * file_name,
                               uint8_t * input_label);

void SbsBaseLayer_getOutputVector (SbsBaseLayer * layer,
                                   float * output_vector,
                                   uint16_t output_vector_size);

void SbsBaseLayer_initialize (SbsBaseLayer * layer);

void SbsBaseLayer_cacheFlush (SbsBaseLayer * layer);

Result SbsBaseLayer_initializeProcessingUnit (SbsBaseLayer * layer);

#ifdef __cplusplus
}
#endif
#endif /* SBS_LAYER_H_ */
