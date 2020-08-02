/*
 * sbs_nn.h
 *
 *  Created on: Sep 7, 2019
 *      Author: Yarib Nevarez
 */

#ifndef SBS_NEURAL_NETWORK_H_
#define SBS_NEURAL_NETWORK_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#include <result.h>
#include "sbs_layer.h"
#include "sbs_weight_matrix.h"
#include "sbs_processing_unit.h"

#pragma pack(push)
#pragma pack(1)

typedef struct SbsNetwork_VTable SbsNetwork;
struct SbsNetwork_VTable
{
  SbsNetwork * (*new)               (void);
  void         (*delete)            (SbsNetwork ** network);
  void         (*giveLayer)         (SbsNetwork * network, SbsLayer * layer);
  void         (*loadInput)         (SbsNetwork * network, char * file_name);
  void         (*updateCycle)       (SbsNetwork * network, uint16_t cycles);
  uint8_t      (*getInferredOutput) (SbsNetwork * network);
  uint8_t      (*getInputLabel)     (SbsNetwork * network);
  /* Note: 'NeuronState ** output_vector' must use intermediate variables
   *  to support unaligned accesses in ARM architectures */
  void         (*getOutputVector)   (SbsNetwork * network,
                                     float * output_vector,
                                     uint16_t output_vector_size);
  void         (*printStatistics)   (SbsNetwork * network);
};
extern struct SbsNetwork_VTable _SbsNetwork;

typedef struct
{
  SbsNetwork *    (*Network)(void);

  SbsLayer *      (*Layer)  (SbsLayerType layer_type,
                             uint16_t rows,
                             uint16_t columns,
                             uint16_t neurons,
                             uint16_t kernel_size,
                             uint16_t kernel_stride,
                             WeightShift weight_shift);

  SbsWeightMatrix (*WeightMatrix)(uint16_t rows,
                                  uint16_t columns,
                                  uint16_t spikes,
                                  uint16_t neurons,
                                  size_t memory_padding,
                                  char * file_name);

  SbsLayer *      (*InputLayer)  (SbsLayerType layer_type,
                                  uint16_t rows,
                                  uint16_t columns,
                                  uint16_t neurons);

  SbsLayer *      (*ConvolutionLayer)(SbsLayerType layer_type,
                                      uint16_t rows,
                                      uint16_t columns,
                                      uint16_t neurons,
                                      uint16_t kernel_size,
                                      WeightShift weight_shift);

  SbsLayer *      (*PoolingLayer)(SbsLayerType layer_type,
                                  uint16_t rows,
                                  uint16_t columns,
                                  uint16_t neurons,
                                  uint16_t kernel_size,
                                  WeightShift weight_shift);

  SbsLayer *      (*FullyConnectedLayer)(SbsLayerType layer_type,
                                         uint16_t neurons,
                                         uint16_t kernel_size,
                                         WeightShift weight_shift);

  SbsLayer *      (*OutputLayer)(SbsLayerType layer_type,
                                 uint16_t neurons,
                                 WeightShift weight_shift);
} SbsNew;

extern SbsNew sbs_new;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif
#endif /* SBS_NEURAL_NETWORK_H_ */
