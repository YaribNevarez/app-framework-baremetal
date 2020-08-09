/*
 * sbs_conv_layer_64.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_conv_layer_64.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsHardware_convLayer64_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsHardware_convLayer64_new (void)
{
  return malloc (sizeof(XSbs_conv_layer_64));
}

static uint32_t  SbsHardware_convLayer64_InterruptSetHandler (void *instance,
                                                             uint32_t ID,
                                                             ARM_GIC_InterruptHandler handler,
                                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsHardware SbsHardware_convLayer64 =
{
  .new =    SbsHardware_convLayer64_new,
  .delete = SbsHardware_convLayer64_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_conv_layer_64_Initialize,
  .Start =              (void (*)(void *))      XSbs_conv_layer_64_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_conv_layer_64_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_conv_layer_64_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_conv_layer_64_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_conv_layer_64_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_conv_layer_64_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_conv_layer_64_Get_return,

  .Set_mode =       (void (*) (void *, SbsHwMode))  XSbs_conv_layer_64_Set_mode,
  .Get_mode =       (uint32_t(*) (void *))          XSbs_conv_layer_64_Get_mode,
  .Set_flags =      (void (*) (void *, uint32_t ))  XSbs_conv_layer_64_Set_flags,
  .Get_flags =      (uint32_t(*) (void *))          XSbs_conv_layer_64_Get_flags,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  NULL,
  .Get_layerSize =  (uint32_t(*) (void *))          NULL,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  NULL,
  .Get_kernelSize = (uint32_t(*) (void *))          NULL,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  NULL,
  .Get_vectorSize = (uint32_t(*) (void *))          NULL,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  NULL,
  .Get_epsilon =    (uint32_t(*) (void *))          NULL,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_conv_layer_64_Set_debug,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_conv_layer_64_Get_debug,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_conv_layer_64_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_conv_layer_64_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_conv_layer_64_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_conv_layer_64_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_conv_layer_64_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_conv_layer_64_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_conv_layer_64_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_convLayer64_InterruptSetHandler
};
