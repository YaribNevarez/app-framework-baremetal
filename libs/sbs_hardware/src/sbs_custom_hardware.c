/*
 * sbs_hardware_update.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_custom_hardware.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsHardware_custom_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsHardware_custom_new (void)
{
  return malloc (sizeof(XSbs_accelerator_64));
}

static uint32_t  SbsHardware_custom_InterruptSetHandler (void *instance,
                                                             uint32_t ID,
                                                             ARM_GIC_InterruptHandler handler,
                                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsHardware SbsHardware_custom =
{
  .new =    SbsHardware_custom_new,
  .delete = SbsHardware_custom_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_accelerator_64_Initialize,
  .Start =              (void (*)(void *))      XSbs_accelerator_64_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_accelerator_64_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_accelerator_64_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_accelerator_64_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_accelerator_64_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_accelerator_64_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_accelerator_64_Get_return,

  .Set_mode =       (void (*) (void *, uint32_t ))  NULL,
  .Get_mode =       (uint32_t(*) (void *))          NULL,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  XSbs_accelerator_64_Set_layerSize,
  .Get_layerSize =  (uint32_t(*) (void *))          XSbs_accelerator_64_Get_layerSize,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  XSbs_accelerator_64_Set_kernelSize,
  .Get_kernelSize = (uint32_t(*) (void *))          XSbs_accelerator_64_Get_kernelSize,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  XSbs_accelerator_64_Set_vectorSize,
  .Get_vectorSize = (uint32_t(*) (void *))          XSbs_accelerator_64_Get_vectorSize,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  XSbs_accelerator_64_Set_epsilon,
  .Get_epsilon =    (uint32_t(*) (void *))          XSbs_accelerator_64_Get_epsilon,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_accelerator_64_Set_debug_r,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_accelerator_64_Get_debug_r,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_accelerator_64_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_accelerator_64_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_accelerator_64_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_accelerator_64_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_accelerator_64_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_accelerator_64_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_accelerator_64_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_custom_InterruptSetHandler
};
