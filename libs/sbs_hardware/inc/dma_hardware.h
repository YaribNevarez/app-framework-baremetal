/*
 * dma_hardware.h
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
#ifndef SBS_DMA_HARDWARE_H_
#define SBS_DMA_HARDWARE_H_

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#include <stdint.h>
#include <stddef.h>

#include <result.h>

#include "xaxidma.h"
#include "xil_types.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/
typedef enum
{
  MEMORY_TO_HARDWARE = XAXIDMA_DMA_TO_DEVICE,
  HARDWARE_TO_MEMORY = XAXIDMA_DEVICE_TO_DMA
} DMATransferDirection;

typedef struct
{
  void *    (*new)    (void);
  void      (*delete) (void ** InstancePtr);

  uint32_t  (*Move)   (void * InstancePtr, void * BuffAddr, uint32_t Length, int Direction)
} DMAHardware;
/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SBS_DMA_HARDWARE_H_ */
