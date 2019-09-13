/*
 * sbs_nn.h
 *
 *  Created on: Sep 7, 2019
 *      Author: yarib
 */

#ifndef SBS_NN_H_
#define SBS_NN_H_

#include <stdint.h>

void sbs_test(void);


typedef enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
} WeightShift;

typedef float  NeuronState;

typedef void * SbsLayer;
typedef void * SbsInputLayer;
typedef void * SbsConvolutionLayer;
typedef void * SbsPoolingLayer;
typedef void * SbsFullyConnectedLayer;
typedef void * SbsOutputLayer;
typedef void * SbsWeightMatrix;

typedef struct SbsNetwork_VTable SbsNetwork;
struct SbsNetwork_VTable
{
  SbsNetwork * (*new)               (void);
  void         (*delete)            (SbsNetwork ** network);
  void         (*giveLayer)         (SbsNetwork * network, SbsLayer layer);
  void         (*loadInput)         (SbsNetwork * network, char * file_name);
  void         (*updateCycle)       (SbsNetwork * network, uint16_t cycles);
  uint8_t      (*getInferredOutput) (SbsNetwork * network);
  uint8_t      (*getInputLabel)     (SbsNetwork * network);
  void         (*getOutputVector)   (SbsNetwork * network, NeuronState ** output_vector, uint16_t * output_vector_size);
};
extern struct SbsNetwork_VTable SbsNetwork_vtable;


//typedef struct SbsWeightMatrix_VTable SbsWeightMatrix;
//struct SbsWeightMatrix_VTable
//{
//  SbsWeightMatrix * (*new)(uint16_t rows, uint16_t columns, char * file_name);
//};
//extern struct SbsWeightMatrix_VTable SbsWeightMatrix_vtable;

SbsWeightMatrix SbsWeightMatrix_new(uint16_t rows, uint16_t columns, char * file_name);

SbsInputLayer SbsInputLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons);

SbsConvolutionLayer SbsConvolutionLayer_new(uint16_t rows,
                                            uint16_t columns,
                                            uint16_t neurons,
                                            uint16_t kernel_size,
                                            WeightShift weight_shift,
                                            uint16_t neurons_prev_Layer);

SbsPoolingLayer SbsPoolingLayer_new(uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift,
                                    uint16_t neurons_prev_Layer);

SbsFullyConnectedLayer SbsFullyConnectedLayer_new(uint16_t neurons,
                                                  uint16_t kernel_size,
                                                  WeightShift weight_shift,
                                                  uint16_t neurons_prev_Layer);

SbsOutputLayer SbsOutputLayer_new(uint16_t neurons,
                                  WeightShift weight_shift,
                                  uint16_t neurons_prev_Layer);

void SbsBaseLayer_setEpsilon(void * layer, float epsilon);
void SbsBaseLayer_giveWeights(void * layer, void * weight_matrix);

#endif /* SBS_NN_H_ */
