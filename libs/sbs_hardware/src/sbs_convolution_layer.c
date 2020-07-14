/*
 * sbs_convolution_layer.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_convolution_layer.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void SbsHardware_convolutionLayer_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}


static void * SbsHardware_convolutionLayer_new (void)
{
  return malloc (sizeof(XSbs_convolution_layer));
}

static uint32_t  SbsHardware_convolutionLayer_InterruptSetHandler (void *instance,
                                                             uint32_t ID,
                                                             ARM_GIC_InterruptHandler handler,
                                                             void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

SbsHardware SbsHardware_convolutionLayer =
{
  .new =    SbsHardware_convolutionLayer_new,
  .delete = SbsHardware_convolutionLayer_delete,

  .Initialize =         (int (*)(void *, u16))  XSbs_convolution_layer_Initialize,
  .Start =              (void (*)(void *))      XSbs_convolution_layer_Start,
  .IsDone =             (uint32_t(*)(void *))   XSbs_convolution_layer_IsDone,
  .IsIdle =             (uint32_t(*) (void *))  XSbs_convolution_layer_IsIdle,
  .IsReady =            (uint32_t(*) (void *))  XSbs_convolution_layer_IsReady,
  .EnableAutoRestart =  (void (*) (void *))     XSbs_convolution_layer_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *))     XSbs_convolution_layer_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XSbs_convolution_layer_Get_return,

  .Set_mode =       (void (*) (void *, SbsHwMode))  XSbs_convolution_layer_Set_mode,
  .Get_mode =       (uint32_t(*) (void *))          XSbs_convolution_layer_Get_mode,
  .Set_layerSize =  (void (*) (void *, uint32_t ))  NULL,
  .Get_layerSize =  (uint32_t(*) (void *))          NULL,
  .Set_kernelSize = (void (*) (void *, uint32_t ))  NULL,
  .Get_kernelSize = (uint32_t(*) (void *))          NULL,
  .Set_vectorSize = (void (*) (void *, uint32_t ))  NULL,
  .Get_vectorSize = (uint32_t(*) (void *))          NULL,
  .Set_epsilon =    (void (*) (void *, uint32_t ))  NULL,
  .Get_epsilon =    (uint32_t(*) (void *))          NULL,
  .Set_debug =      (void (*) (void *, uint32_t ))  XSbs_convolution_layer_Set_debug_r,
  .Get_debug =      (uint32_t(*) (void *))          XSbs_convolution_layer_Get_debug_r,

  .InterruptGlobalEnable =  (void (*) (void *))             XSbs_convolution_layer_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *))             XSbs_convolution_layer_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t ))  XSbs_convolution_layer_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t ))  XSbs_convolution_layer_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t ))  XSbs_convolution_layer_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *))          XSbs_convolution_layer_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *))          XSbs_convolution_layer_InterruptGetStatus,

  .InterruptSetHandler = SbsHardware_convolutionLayer_InterruptSetHandler
};
