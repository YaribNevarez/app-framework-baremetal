/*
 * sbs_hardware_update.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_hardware_update.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsHardware_fixedpoint_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsHardware_fixedpoint_new (void)
{
  return malloc (sizeof(XSbs_accelerator));
}

static uint32_t  SbsHardware_fixedpoint_InterruptSetHandler (void *instance,
                                                             uint32_t ID,
                                                             ARM_GIC_InterruptHandler handler,
                                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsHardware SbsHardware_fixedpoint =
{
  .new =    SbsHardware_fixedpoint_new,
  .delete = SbsHardware_fixedpoint_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_accelerator_Initialize,
  .Start =              (void (*)(void *))      XSbs_accelerator_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_accelerator_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_accelerator_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_accelerator_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_accelerator_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_accelerator_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  NULL,

  .Set_mode =       (void (*) (void *, uint32_t ))  NULL,
  .Get_mode =       (uint32_t(*) (void *))          NULL,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  XSbs_accelerator_Set_layerSize,
  .Get_layerSize =  (uint32_t(*) (void *))          XSbs_accelerator_Get_layerSize,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  XSbs_accelerator_Set_kernelSize,
  .Get_kernelSize = (uint32_t(*) (void *))          XSbs_accelerator_Get_kernelSize,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  XSbs_accelerator_Set_vectorSize,
  .Get_vectorSize = (uint32_t(*) (void *))          XSbs_accelerator_Get_vectorSize,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  XSbs_accelerator_Set_epsilon,
  .Get_epsilon =    (uint32_t(*) (void *))          XSbs_accelerator_Get_epsilon,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_accelerator_Set_debug,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_accelerator_Get_debug,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_accelerator_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_accelerator_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_accelerator_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_accelerator_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_accelerator_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_accelerator_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_accelerator_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_fixedpoint_InterruptSetHandler
};
