/*
 * sbs_conv_layer_32.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_conv_layer_32.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsHardware_convLayer32_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsHardware_convLayer32_new (void)
{
  return malloc (sizeof(XSbs_conv_layer_32));
}

static uint32_t  SbsHardware_convLayer32_InterruptSetHandler (void *instance,
                                                             uint32_t ID,
                                                             ARM_GIC_InterruptHandler handler,
                                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsHardware SbsHardware_convLayer32 =
{
  .new =    SbsHardware_convLayer32_new,
  .delete = SbsHardware_convLayer32_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_conv_layer_32_Initialize,
  .Start =              (void (*)(void *))      XSbs_conv_layer_32_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_conv_layer_32_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_conv_layer_32_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_conv_layer_32_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_conv_layer_32_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_conv_layer_32_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_conv_layer_32_Get_return,

  .Set_mode =       (void (*) (void *, SbsHwMode))  XSbs_conv_layer_32_Set_mode,
  .Get_mode =       (uint32_t(*) (void *))          XSbs_conv_layer_32_Get_mode,
  .Set_flags =      (void (*) (void *, uint32_t ))  XSbs_conv_layer_32_Set_flags,
  .Get_flags =      (uint32_t(*) (void *))          XSbs_conv_layer_32_Get_flags,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  NULL,
  .Get_layerSize =  (uint32_t(*) (void *))          NULL,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  NULL,
  .Get_kernelSize = (uint32_t(*) (void *))          NULL,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  NULL,
  .Get_vectorSize = (uint32_t(*) (void *))          NULL,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  NULL,
  .Get_epsilon =    (uint32_t(*) (void *))          NULL,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_conv_layer_32_Set_debug,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_conv_layer_32_Get_debug,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_conv_layer_32_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_conv_layer_32_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_conv_layer_32_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_conv_layer_32_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_conv_layer_32_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_conv_layer_32_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_conv_layer_32_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_convLayer32_InterruptSetHandler
};
