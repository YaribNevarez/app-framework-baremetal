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
#include <gic.h>

//#include "xaxidma.h"
#include "xil_types.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/
typedef enum
{
  MEMORY_TO_HARDWARE,
  HARDWARE_TO_MEMORY
} DMATransferDirection;

typedef enum
{
  DMA_IRQ_IOC,   /* Completion */
  DMA_IRQ_DELAY, /* Delay      */
  DMA_IRQ_ERROR, /* Error      */
  DMA_IRQ_ALL    /* All        */
} DMAIRQMask;

typedef struct
{
  void *    (*new)              (void);
  void      (*delete)           (void ** instance_ptr);

  int       (*Initialize)       (void * instance, u16 deviceId);

  uint32_t  (*Move)             (void * instance,
                                 void * BuffAddr,
                                 uint32_t Length,
                                 DMATransferDirection direction);

  void      (*InterruptEnable)  (void * instance,
                                 DMAIRQMask mask,
                                 DMATransferDirection direction);

  void      (*InterruptDisable) (void * instance,
                                 DMAIRQMask mask,
                                 DMATransferDirection direction);

  void        (*InterruptClear)       (void * instance,
                                       DMAIRQMask mask,
                                       DMATransferDirection direction);

  DMAIRQMask  (*InterruptGetEnabled)  (void * instance,
                                       DMATransferDirection direction);

  DMAIRQMask  (*InterruptGetStatus)   (void * instance,
                                       DMATransferDirection direction);

  void        (*Reset)                (void * instance);

  int         (*ResetIsDone)          (void * instance);

  uint32_t    (*InterruptSetHandler)  (void *InstancePtr,
                                       uint32_t ID,
                                       ARM_GIC_InterruptHandler handler,
                                       void * data);
} DMAHardware;
/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SBS_DMA_HARDWARE_H_ */
