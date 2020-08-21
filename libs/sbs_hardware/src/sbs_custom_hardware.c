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
  return malloc (sizeof(XSbs_accelerator_unit));
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

  .Initialize =         (int (*)(void *, u16))  XSbs_accelerator_unit_Initialize,
  .Start =              (void (*)(void *))      XSbs_accelerator_unit_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_accelerator_unit_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_accelerator_unit_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_accelerator_unit_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_accelerator_unit_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_accelerator_unit_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_accelerator_unit_Get_return,

  .Set_mode =       (void (*) (void *, SbsHwMode )) NULL,
  .Get_mode =       (uint32_t(*) (void *))          NULL,
  .Set_flags =      (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_Set_flags,
  .Get_flags =      (uint32_t(*) (void *))          XSbs_accelerator_unit_Get_flags,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_Set_layerSize,
  .Get_layerSize =  (uint32_t(*) (void *))          XSbs_accelerator_unit_Get_layerSize,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_Set_kernelSize,
  .Get_kernelSize = (uint32_t(*) (void *))          XSbs_accelerator_unit_Get_kernelSize,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_Set_vectorSize,
  .Get_vectorSize = (uint32_t(*) (void *))          XSbs_accelerator_unit_Get_vectorSize,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_Set_epsilon,
  .Get_epsilon =    (uint32_t(*) (void *))          XSbs_accelerator_unit_Get_epsilon,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_Set_debug,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_accelerator_unit_Get_debug,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_accelerator_unit_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_accelerator_unit_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_accelerator_unit_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_accelerator_unit_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_accelerator_unit_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_custom_InterruptSetHandler
};
