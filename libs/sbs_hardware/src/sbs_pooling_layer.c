/*
 * sbs_hardware_update.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_pooling_layer.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsHardware_poolingLayer_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsHardware_poolingLayer_new (void)
{
  return malloc (sizeof(XSbs_pooling_layer));
}

static uint32_t  SbsHardware_poolingLayer_InterruptSetHandler (void *instance,
                                                             uint32_t ID,
                                                             ARM_GIC_InterruptHandler handler,
                                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsHardware SbsHardware_poolingLayer =
{
  .new =    SbsHardware_poolingLayer_new,
  .delete = SbsHardware_poolingLayer_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_pooling_layer_Initialize,
  .Start =              (void (*)(void *))      XSbs_pooling_layer_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_pooling_layer_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_pooling_layer_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_pooling_layer_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_pooling_layer_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_pooling_layer_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_pooling_layer_Get_return,

  .Set_mode =       (void (*) (void *, SbsHwMode )) NULL,
  .Get_mode =       (uint32_t(*) (void *))          NULL,
  .Set_flags =      (void (*) (void *, uint32_t ))  NULL,
  .Get_flags =      (uint32_t(*) (void *))          NULL,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  XSbs_pooling_layer_Set_layerSize,
  .Get_layerSize =  (uint32_t(*) (void *))          XSbs_pooling_layer_Get_layerSize,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  XSbs_pooling_layer_Set_kernelSize,
  .Get_kernelSize = (uint32_t(*) (void *))          XSbs_pooling_layer_Get_kernelSize,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  XSbs_pooling_layer_Set_vectorSize,
  .Get_vectorSize = (uint32_t(*) (void *))          XSbs_pooling_layer_Get_vectorSize,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  XSbs_pooling_layer_Set_epsilon,
  .Get_epsilon =    (uint32_t(*) (void *))          XSbs_pooling_layer_Get_epsilon,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_pooling_layer_Set_debug,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_pooling_layer_Get_debug,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_pooling_layer_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_pooling_layer_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_pooling_layer_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_pooling_layer_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_pooling_layer_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_pooling_layer_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_pooling_layer_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_poolingLayer_InterruptSetHandler
};
