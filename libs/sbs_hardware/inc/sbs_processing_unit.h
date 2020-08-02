/*
 * sbs_processing_unit.h
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
#ifndef SBS_PROCESSING_UNIT_H_
#define SBS_PROCESSING_UNIT_H_

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#include <stdint.h>
#include <stddef.h>

#include <result.h>
#include "sbs_hardware.h"
#include "dma_hardware.h"
#include "multivector.h"

#include "memory_manager.h"
#include "timer.h"
#include "event.h"

#include "xil_types.h"
/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

typedef enum
{
  NONE_LAYER               = 0,
  HX_INPUT_LAYER           = 1<<0,
  H1_CONVOLUTION_LAYER     = 1<<1,
  H2_POOLING_LAYER         = 1<<2,
  H3_CONVOLUTION_LAYER     = 1<<3,
  H4_POOLING_LAYER         = 1<<4,
  H5_FULLY_CONNECTED_LAYER = 1<<5,
  HY_OUTPUT_LAYER          = 1<<6
} SbsLayerType;

typedef enum
{
  SBS_HW_INPUT_LAYER,
  SBS_HW_CONVOLUTION_LAYER,
  SBS_HW_POOLING_LAYER,
  SBS_HW_DENSE_LAYER,
  SBS_HW_OUTPUT_LAYER
} SbSHardwareType;

typedef enum
{
  MEM_CMD_NONE = 0,
  MEM_CMD_COPY,
  MEM_CMD_MOVE,
  MEM_CMD_CLEAR
} MemoryCmdID;

typedef struct
{
  void *      src;
  void *      dest;
  size_t      size;
  MemoryCmdID cmdID;
} MemoryCmd;

typedef struct
{
  uint32_t  layerSize;
  uint32_t  kernelSize;
  uint32_t  vectorSize;
  uint32_t  epsilon;

  size_t    stateBufferSize;
  size_t    weightBufferSize;
  size_t    spikeBufferSize;

  size_t    randBufferPaddingSize;
  size_t    stateBufferPaddingSize;
  size_t    weightBufferPaddingSize;
  size_t    spikeBufferPaddingSize;
  size_t    spikeBatchBufferPaddingSize;

  void *    txBuffer;
  size_t    txBufferSize;

  void *    rxBuffer;
  size_t    rxBufferSize;

  MemoryCmd memory_cmd;

  Event *   event;
} SbsAcceleratorProfie;

typedef struct
{
  SbsHardware *       hwDriver;
  DMAHardware *       dmaDriver;
  uint32_t            layerAssign;
  uint32_t            hwDeviceID;
  uint32_t            dmaDeviceID;
  uint32_t            hwIntVecID;
  uint32_t            dmaTxIntVecID;
  uint32_t            dmaRxIntVecID;
  SbSHardwareType     hwType;
  size_t              channelSize;
  SbsAcceleratorProfie * profile;
  MemoryBlock         ddrMem;
} SbSHardwareConfig;

typedef struct
{
  SbSHardwareConfig *     hardwareConfig;
  void *                  updateHardware;
  void *                  dmaHardware;
  SbsAcceleratorProfie *  profile;

#ifdef DEBUG
  uint32_t    txStateCounter;
  uint32_t    txWeightCounter;
#endif

  uint32_t    txSpikeCounter;

  void *      txBufferCurrentPtr;
  void *      txBuffer;
  size_t      txBufferSize;

  void *      rxBuffer;
  size_t      rxBufferSize;

  /*Below used by hardware interruption*/
  uint32_t    errorFlags;
  uint32_t    txDone;
  uint32_t    rxDone;
  uint32_t    acceleratorReady;
  MemoryCmd   memory_cmd;
} SbSUpdateAccelerator;
/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/
int SbSUpdateAccelerator_getGroupFromList (SbsLayerType layerType,
                                           SbSUpdateAccelerator ** sub_list,
                                           int sub_list_size);

int Accelerator_initialize (SbSUpdateAccelerator * accelerator,
                            SbSHardwareConfig * hardware_config);

void Accelerator_shutdown (SbSUpdateAccelerator * accelerator);

SbSUpdateAccelerator * Accelerator_new (SbSHardwareConfig * hardware_config);

void Accelerator_delete (SbSUpdateAccelerator ** accelerator);

Result Accelerator_loadCoefficients (SbSUpdateAccelerator * accelerator,
                                     SbsAcceleratorProfie * profile,
                                     Multivector * weight_matrix,
                                     int row_vector);

void Accelerator_setup (SbSUpdateAccelerator * accelerator,
                        SbsAcceleratorProfie * profile);

void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                  uint32_t * state_vector);

void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                   uint8_t * weight_vector);

void Accelerator_giveSpike (SbSUpdateAccelerator * accelerator, uint16_t spike);

int Accelerator_start (SbSUpdateAccelerator * accelerator);

SbsAcceleratorProfie * SbsAcceleratorProfie_new (SbsLayerType layerType,
                                                 Multivector * state_matrix,
                                                 Multivector * weight_matrix,
                                                 Multivector * spike_matrix,
                                                 uint32_t kernel_size,
                                                 float epsilon,
                                                 MemoryCmd memory_cmd,
                                                 Event * parent_event);

void SbsAcceleratorProfie_delete (SbsAcceleratorProfie ** profile);

Result SbsPlatform_initialize (SbSHardwareConfig * hardware_config_list,
                               uint32_t list_length,
                               uint32_t MT19937_seed);

void SbsPlatform_shutdown (void);

char * SbsLayerType_string(SbsLayerType layerType);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* SBS_PROCESSING_UNIT_H_ */
