/*
 * sbs_hardware_update.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_dma.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsDMA_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsDMA_new (void)
{
  return malloc (sizeof(XSbs_dma));
}

static uint32_t  SbsDMA_InterruptSetHandler (void *instance,
                                             uint32_t ID,
                                             ARM_GIC_InterruptHandler handler,
                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsDMA SbsDMA_driver =
{
  .new =    SbsDMA_new,
  .delete = SbsDMA_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_dma_Initialize,
  .Start =              (void (*)(void *))      XSbs_dma_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_dma_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_dma_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_dma_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_dma_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_dma_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_dma_Get_return,

  .Set_state_matrix_data = (void (*) (void *, uint32_t)) XSbs_dma_Set_state_matrix_data,
  .Get_state_matrix_data = (uint32_t (*)(void *)) XSbs_dma_Get_state_matrix_data,
  .Set_weight_matrix_data = (void (*) (void *, uint32_t)) XSbs_dma_Set_weight_matrix_data,
  .Get_weight_matrix_data = (uint32_t (*)(void *)) XSbs_dma_Get_weight_matrix_data,
  .Set_input_spike_matrix_data = (void (*) (void *, uint32_t)) XSbs_dma_Set_input_spike_matrix_data,
  .Get_input_spike_matrix_data = (uint32_t (*)(void *)) XSbs_dma_Get_input_spike_matrix_data,
  .Set_output_spike_matrix_data = (void (*) (void *, uint32_t)) XSbs_dma_Set_output_spike_matrix_data,
  .Get_output_spike_matrix_data = (uint32_t (*)(void *)) XSbs_dma_Get_output_spike_matrix_data,
  .Set_debug = (void (*) (void *, uint32_t)) XSbs_dma_Set_debug,
  .Get_debug = (uint32_t (*)(void *)) XSbs_dma_Get_debug,
  .Set_buffer_r = (void (*) (void *, uint32_t)) XSbs_dma_Set_buffer_r,
  .Get_buffer_r = (uint32_t (*)(void *)) XSbs_dma_Get_buffer_r,
  .Set_weight_spikes = (void (*) (void *, uint32_t)) XSbs_dma_Set_weight_spikes,
  .Get_weight_spikes = (uint32_t (*)(void *)) XSbs_dma_Get_weight_spikes,
  .Set_rows = (void (*) (void *, uint32_t)) XSbs_dma_Set_rows,
  .Get_rows = (uint32_t (*)(void *)) XSbs_dma_Get_rows,
  .Set_input_spike_matrix_columns = (void (*) (void *, uint32_t)) XSbs_dma_Set_input_spike_matrix_columns,
  .Get_input_spike_matrix_columns = (uint32_t (*)(void *)) XSbs_dma_Get_input_spike_matrix_columns,
  .Set_input_spike_matrix_rows = (void (*) (void *, uint32_t)) XSbs_dma_Set_input_spike_matrix_rows,
  .Get_input_spike_matrix_rows = (uint32_t (*)(void *)) XSbs_dma_Get_input_spike_matrix_rows,
  .Set_kernel_row_pos = (void (*) (void *, uint32_t)) XSbs_dma_Set_kernel_row_pos,
  .Get_kernel_row_pos = (uint32_t (*)(void *)) XSbs_dma_Get_kernel_row_pos,
  .Set_columns = (void (*) (void *, uint32_t)) XSbs_dma_Set_columns,
  .Get_columns = (uint32_t (*)(void *)) XSbs_dma_Get_columns,
  .Set_vector_size = (void (*) (void *, uint32_t)) XSbs_dma_Set_vector_size,
  .Get_vector_size = (uint32_t (*)(void *)) XSbs_dma_Get_vector_size,
  .Set_kernel_stride = (void (*) (void *, uint32_t)) XSbs_dma_Set_kernel_stride,
  .Get_kernel_stride = (uint32_t (*)(void *)) XSbs_dma_Get_kernel_stride,
  .Set_kernel_size = (void (*) (void *, uint32_t)) XSbs_dma_Set_kernel_size,
  .Get_kernel_size = (uint32_t (*)(void *)) XSbs_dma_Get_kernel_size,
  .Set_layer_weight_shift = (void (*) (void *, uint32_t)) XSbs_dma_Set_layer_weight_shift,
  .Get_layer_weight_shift = (uint32_t (*)(void *)) XSbs_dma_Get_layer_weight_shift,
  .Set_mt19937 = (void (*) (void *, uint32_t)) XSbs_dma_Set_mt19937,
  .Get_mt19937 = (uint32_t (*)(void *)) XSbs_dma_Get_mt19937,
  .Set_epsilon = (void (*) (void *, uint32_t)) XSbs_dma_Set_epsilon,
  .Get_epsilon = (uint32_t (*)(void *)) XSbs_dma_Get_epsilon,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_dma_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_dma_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_dma_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_dma_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_dma_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_dma_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_dma_InterruptGetStatus,

  .InterruptSetHandler = SbsDMA_InterruptSetHandler
};
