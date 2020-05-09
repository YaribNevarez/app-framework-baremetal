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

  .Set_state_matrix_data = (void (*) (void *, uint32_t)) XSbs_dma_Set_state_matrix_data_V,
  .Get_state_matrix_data = (uint32_t (*)(void *)) XSbs_dma_Get_state_matrix_data_V,
  .Set_weight_matrix_data = (void (*) (void *, uint32_t)) XSbs_dma_Set_weight_matrix_data_V,
  .Get_weight_matrix_data = (uint32_t (*)(void *)) XSbs_dma_Get_weight_matrix_data_V,
  .Set_debug = (void (*) (void *, uint32_t)) XSbs_dma_Set_debug_V,
  .Get_debug = (uint32_t (*)(void *)) XSbs_dma_Get_debug_V,
  .Set_buffer_r = (void (*) (void *, uint32_t)) XSbs_dma_Set_buffer_V,
  .Get_buffer_r = (uint32_t (*)(void *)) XSbs_dma_Get_buffer_V,
  .Set_weight_spikes = (void (*) (void *, uint32_t)) XSbs_dma_Set_weight_spikes_V,
  .Get_weight_spikes = (uint32_t (*)(void *)) XSbs_dma_Get_weight_spikes_V,
  .Set_rows = (void (*) (void *, uint32_t)) XSbs_dma_Set_rows_V,
  .Get_rows = (uint32_t (*)(void *)) XSbs_dma_Get_rows_V,
  .Set_input_spike_matrix_columns = (void (*) (void *, uint32_t)) XSbs_dma_Set_input_spike_matrix_columns_V,
  .Get_input_spike_matrix_columns = (uint32_t (*)(void *)) XSbs_dma_Get_input_spike_matrix_columns_V,
  .Set_input_spike_matrix_rows = (void (*) (void *, uint32_t)) XSbs_dma_Set_input_spike_matrix_rows_V,
  .Get_input_spike_matrix_rows = (uint32_t (*)(void *)) XSbs_dma_Get_input_spike_matrix_rows_V,
  .Set_kernel_row_pos = (void (*) (void *, uint32_t)) XSbs_dma_Set_kernel_row_pos_V,
  .Get_kernel_row_pos = (uint32_t (*)(void *)) XSbs_dma_Get_kernel_row_pos_V,
  .Set_columns = (void (*) (void *, uint32_t)) XSbs_dma_Set_columns_V,
  .Get_columns = (uint32_t (*)(void *)) XSbs_dma_Get_columns_V,
  .Set_vector_size = (void (*) (void *, uint32_t)) XSbs_dma_Set_vector_size_V,
  .Get_vector_size = (uint32_t (*)(void *)) XSbs_dma_Get_vector_size_V,
  .Set_kernel_stride = (void (*) (void *, uint32_t)) XSbs_dma_Set_kernel_stride_V,
  .Get_kernel_stride = (uint32_t (*)(void *)) XSbs_dma_Get_kernel_stride_V,
  .Set_kernel_size = (void (*) (void *, uint32_t)) XSbs_dma_Set_kernel_size_V,
  .Get_kernel_size = (uint32_t (*)(void *)) XSbs_dma_Get_kernel_size_V,
  .Set_layer_weight_shift = (void (*) (void *, uint32_t)) XSbs_dma_Set_layer_weight_shift_V,
  .Get_layer_weight_shift = (uint32_t (*)(void *)) XSbs_dma_Get_layer_weight_shift_V,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_dma_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_dma_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_dma_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_dma_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_dma_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_dma_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_dma_InterruptGetStatus,

  .InterruptSetHandler = SbsDMA_InterruptSetHandler
};
