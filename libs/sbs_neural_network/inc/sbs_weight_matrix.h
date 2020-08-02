/*
 * sbs_weight_matrix.h
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */

#ifndef SBS_WEIGHT_MATRIX_H_
#define SBS_WEIGHT_MATRIX_H_
#ifdef __cplusplus
extern "C" {
#endif

#include "xil_types.h"

typedef void * SbsWeightMatrix;

SbsWeightMatrix SbsWeightMatrix_new (uint16_t rows,
                                     uint16_t columns,
                                     uint16_t spikes,
                                     uint16_t neurons,
                                     size_t memory_padding,
                                     char * file_name);

#ifdef __cplusplus
}
#endif
#endif /* SBS_WEIGHT_MATRIX_H_ */
