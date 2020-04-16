/*
 * sbs_spike_master.h
 *
 *  Created on: April 15, 2020
 *      Author: Yarib Nevarez
 */

#ifndef SBS_SPIKE_MASTER_H_
#define SBS_SPIKE_MASTER_H_
#ifdef __cplusplus
extern "C" {
#endif

void sbs_spike_master (unsigned int * spike_matrix_mem,
                       unsigned int * state_matrix_mem,
                       unsigned int rows,
                       unsigned int columns,
                       unsigned int vector_size,
                       unsigned int seed,
                       unsigned int * debug_mem);

#ifdef __cplusplus
}
#endif
#endif /* SBS_SPIKE_MASTER_H_ */
