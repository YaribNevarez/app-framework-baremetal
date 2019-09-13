/*
 * sbs_nn.h
 *
 *  Created on: Sep 7, 2019
 *      Author: yarib
 */

#ifndef SBS_NN_H_
#define SBS_NN_H_

#include <stdint.h>

typedef enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
} WeightShift;

typedef float  NeuronState;
typedef void * SbsWeightMatrix;

typedef struct SbsLayer_VTable SbsLayer;
struct SbsLayer_VTable
{
  SbsLayer * (*new)        (uint16_t rows,
                            uint16_t columns,
                            uint16_t neurons,
                            uint16_t kernel_size,
                            uint16_t kernel_stride,
                            WeightShift weight_shift,
                            uint16_t    neurons_previous_Layer);
  void       (*delete)     (SbsLayer ** layer);
  void       (*setEpsilon) (SbsLayer * layer, float epsilon);
  void       (*giveWeights)(SbsLayer * layer, SbsWeightMatrix weight_matrix);
};
extern struct SbsLayer_VTable _SbsLayer;


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
  void         (*getOutputVector)   (SbsNetwork * network, NeuronState ** output_vector, uint16_t * output_vector_size);
};
extern struct SbsNetwork_VTable _SbsNetwork;

typedef struct
{
  SbsNetwork * (*Network)(void);

  SbsLayer *   (*Layer)  (uint16_t rows,
                          uint16_t columns,
                          uint16_t neurons,
                          uint16_t kernel_size,
                          uint16_t kernel_stride,
                          WeightShift weight_shift,
                          uint16_t    neurons_previous_Layer);

  SbsWeightMatrix (*WeightMatrix)(uint16_t rows, uint16_t columns, char * file_name);

  SbsLayer *      (*InputLayer)  (uint16_t rows, uint16_t columns, uint16_t neurons);

  SbsLayer *      (*ConvolutionLayer)(uint16_t rows,
                                      uint16_t columns,
                                      uint16_t neurons,
                                      uint16_t kernel_size,
                                      WeightShift weight_shift,
                                      uint16_t neurons_prev_Layer);

  SbsLayer * (*PoolingLayer)(uint16_t rows,
                             uint16_t columns,
                             uint16_t neurons,
                             uint16_t kernel_size,
                             WeightShift weight_shift,
                             uint16_t neurons_prev_Layer);

  SbsLayer * (*FullyConnectedLayer)(uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift,
                                    uint16_t neurons_prev_Layer);

  SbsLayer * (*OutputLayer)(uint16_t neurons,
                            WeightShift weight_shift,
                            uint16_t neurons_prev_Layer);
} SbsNew;


extern SbsNew sbs_new;


#endif /* SBS_NN_H_ */
