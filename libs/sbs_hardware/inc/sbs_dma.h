/*
 * sbs_dma.h
 *
 *  Created on: May 3rd, 2020
 *      Author: Yarib Nevarez
 */
#ifndef SBS_DMA_H_
#define SBS_DMA_H_

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#include <stdint.h>
#include <stddef.h>

#include <result.h>
#include "gic.h"

#include "xil_types.h"
#include "xsbs_dma.h"
/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/
typedef struct
{
  void *    (*new)(void);
  void      (*delete)(void ** InstancePtr);

  int       (*Initialize) (void *InstancePtr, u16 deviceId);
  void      (*Start)      (void *InstancePtr);
  uint32_t  (*IsDone)     (void *InstancePtr);
  uint32_t  (*IsIdle)     (void *InstancePtr);
  uint32_t  (*IsReady)    (void *InstancePtr);
  void      (*EnableAutoRestart) (void *InstancePtr);
  void      (*DisableAutoRestart) (void *InstancePtr);
  uint32_t  (*Get_return) (void *InstancePtr);

  void      (*Set_state_matrix_data ) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_state_matrix_data) (void *InstancePtr);
  void      (*Set_weight_matrix_data) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_weight_matrix_data) (void *InstancePtr);
  void      (*Set_debug) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_debug) (void *InstancePtr);
  void      (*Set_buffer_r) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_buffer_r) (void *InstancePtr);
  void      (*Set_weight_spikes) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_weight_spikes) (void *InstancePtr);
  void      (*Set_rows) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_rows) (void *InstancePtr);
  void      (*Set_input_spike_matrix_columns) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_input_spike_matrix_columns) (void *InstancePtr);
  void      (*Set_input_spike_matrix_rows) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_input_spike_matrix_rows) (void *InstancePtr);
  void      (*Set_kernel_row_pos) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_kernel_row_pos) (void *InstancePtr);
  void      (*Set_columns) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_columns) (void *InstancePtr);
  void      (*Set_vector_size) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_vector_size) (void *InstancePtr);
  void      (*Set_kernel_stride) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_kernel_stride) (void *InstancePtr);
  void      (*Set_kernel_size) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_kernel_size) (void *InstancePtr);
  void      (*Set_layer_weight_shift) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_layer_weight_shift) (void *InstancePtr);

  void      (*InterruptGlobalEnable)  (void *InstancePtr);
  void      (*InterruptGlobalDisable) (void *InstancePtr);
  void      (*InterruptEnable)        (void *InstancePtr, uint32_t Mask);
  void      (*InterruptDisable)       (void *InstancePtr, uint32_t Mask);
  void      (*InterruptClear)         (void *InstancePtr, uint32_t Mask);
  uint32_t  (*InterruptGetEnabled)    (void *InstancePtr);
  uint32_t  (*InterruptGetStatus)     (void *InstancePtr);

  uint32_t  (*InterruptSetHandler)    (void *InstancePtr,
                                       uint32_t ID,
                                       ARM_GIC_InterruptHandler handler,
                                       void * data);
} SbsDMA;
/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

extern SbsDMA SbsDMA_driver;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SBS_HARDWARE_UPDATE_H_ */
