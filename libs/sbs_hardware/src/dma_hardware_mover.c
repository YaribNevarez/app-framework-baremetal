/*
 * dma_hardware_mover.h
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "dma_hardware_mover.h"
#include "stdio.h"
#include "stdlib.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

static void * DMAHardware_new (void)
{
  return malloc (sizeof(XAxiDma));
}

static void DMAHardware_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}

DMAHardware DMAHardware_mover =
{
  .new =    DMAHardware_new,
  .delete = DMAHardware_delete,

  .Move =   XAxiDma_SimpleTransfer
};
