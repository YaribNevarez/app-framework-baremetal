/*
 * sbs_hardware.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_processing_unit.h"
#include "sbs_hardware_spike.h"
#include "sbs_hardware_update.h"
#include "dma_hardware_mover.h"
#include "miscellaneous.h"
#include "stdio.h"
#include "xscugic.h"
#include "mt19937int.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

//#define NUM_ACCELERATOR_INSTANCES  (sizeof(SbSHardwareConfig_list) / sizeof(SbSHardwareConfig))

static SbSUpdateAccelerator **  SbSUpdateAccelerator_list = NULL;
static uint8_t SbSUpdateAccelerator_list_length = 0;

int SbSUpdateAccelerator_getGroupFromList (SbsLayerType layerType, SbSUpdateAccelerator ** sub_list, int sub_list_size)
{
  int sub_list_count = 0;
  int i;

  ASSERT (sub_list != NULL);
  ASSERT (0 < sub_list_size);

  ASSERT(SbSUpdateAccelerator_list != NULL);
  for (i = 0; sub_list_count < sub_list_size && i < SbSUpdateAccelerator_list_length;
      i++)
  {
    ASSERT(SbSUpdateAccelerator_list[i] != NULL);
    ASSERT(SbSUpdateAccelerator_list[i]->hardwareConfig != NULL);
    if (SbSUpdateAccelerator_list[i] != NULL
        && SbSUpdateAccelerator_list[i]->hardwareConfig->layerAssign & layerType)
    {
      sub_list[sub_list_count ++] = SbSUpdateAccelerator_list[i];
    }
  }
  return sub_list_count;
}


static XScuGic                 ScuGic = {0};

#define ACCELERATOR_DMA_RESET_TIMEOUT 10000

static void Accelerator_txInterruptHandler(void * data)
{
  XAxiDma *AxiDmaInst = ((SbSUpdateAccelerator *) data)->dmaHardware;
  uint32_t IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DMA_TO_DEVICE);

  XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DMA_TO_DEVICE);

  if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
  {
    return;
  }

  if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
  {
    int TimeOut;

    ((SbSUpdateAccelerator *) data)->errorFlags |= 0x01;

    XAxiDma_Reset (AxiDmaInst);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (XAxiDma_ResetIsDone (AxiDmaInst)) break;

    printf("Possible illegal address access\n");
    ASSERT(0);
    return;
  }

  if (IrqStatus &  XAXIDMA_IRQ_IOC_MASK)
  {
    ((SbSUpdateAccelerator *) data)->txDone = 1;
  }
}

static void Accelerator_rxInterruptHandler (void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  XAxiDma *AxiDmaInst = accelerator->dmaHardware;
  uint32_t IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DEVICE_TO_DMA);

  XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DEVICE_TO_DMA);

  if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
  {
    return;
  }

  if (IrqStatus & XAXIDMA_IRQ_DELAY_MASK)
  {
    return;
  }

  if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
  {
    int TimeOut;

    ((SbSUpdateAccelerator *) data)->errorFlags |= 0x01;

    XAxiDma_Reset (AxiDmaInst);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (XAxiDma_ResetIsDone (AxiDmaInst)) break;

    printf("Possible illegal address access\n");
    ASSERT(0);
    return;
  }

  if ((IrqStatus &  XAXIDMA_IRQ_IOC_MASK))
  {
    Xil_DCacheInvalidateRange((INTPTR)accelerator->rxBuffer, accelerator->rxBufferSize);

    accelerator->txDone = 1;
    accelerator->rxDone = 1;

    if (accelerator->memory_cmd.cmdID == MEM_CMD_MOVE)
      memcpy(accelerator->memory_cmd.dest,
             accelerator->memory_cmd.src,
             accelerator->memory_cmd.size);
  }
}

static void Accelerator_hardwareInterruptHandler (void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  uint32_t status;

  ASSERT (accelerator != NULL);
  ASSERT (accelerator->hardwareConfig != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver->InterruptGetStatus != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver->InterruptClear != NULL);

  status = accelerator->hardwareConfig->hwDriver->InterruptGetStatus(accelerator->updateHardware);
  accelerator->hardwareConfig->hwDriver->InterruptClear(accelerator->updateHardware, status);
  accelerator->acceleratorReady = status & 1;
}

int Accelerator_initialize(SbSUpdateAccelerator * accelerator,
                                  SbSHardwareConfig * hardware_config)
{
  XScuGic_Config *    IntcConfig;
  XAxiDma_Config *    dmaConfig;
  int                 status;

  ASSERT (accelerator != NULL);
  ASSERT (hardware_config != NULL);

  if (accelerator == NULL || hardware_config == NULL)
    return XST_FAILURE;

  memset (accelerator, 0x00, sizeof(SbSUpdateAccelerator));

  accelerator->hardwareConfig = hardware_config;

  /******************************* DMA initialization ************************/

  accelerator->dmaHardware = hardware_config->dmaDriver->new();

  ASSERT(accelerator->dmaHardware != NULL);

  if (accelerator->dmaHardware == NULL)
  {
    xil_printf ("ERROR: DMA instance memory allocation fail\r\n", hardware_config->dmaDeviceID);

    return XST_FAILURE;
  }

  if (hardware_config->mode == HARDWARE)
  {
    dmaConfig = XAxiDma_LookupConfig (hardware_config->dmaDeviceID);
    if (dmaConfig == NULL)
    {
      xil_printf ("No configuration found for %d\r\n",
                  hardware_config->dmaDeviceID);

      return XST_FAILURE;
    }

    status = XAxiDma_CfgInitialize (accelerator->dmaHardware, dmaConfig);
    if (status != XST_SUCCESS)
    {
      xil_printf ("Initialization failed %d\r\n", status);
      return XST_FAILURE;
    }

    if (XAxiDma_HasSg((XAxiDma* )accelerator->dmaHardware))
    {
      xil_printf ("Device configured as SG mode \r\n");

      return XST_FAILURE;
    }

    if (hardware_config->dmaTxIntVecID)
      XAxiDma_IntrEnable((XAxiDma* )accelerator->dmaHardware,
                         XAXIDMA_IRQ_ALL_MASK,
                         XAXIDMA_DMA_TO_DEVICE);

    else
      XAxiDma_IntrDisable((XAxiDma* )accelerator->dmaHardware,
                          XAXIDMA_IRQ_ALL_MASK,
                          XAXIDMA_DMA_TO_DEVICE);

    if (hardware_config->dmaRxIntVecID)
      XAxiDma_IntrEnable((XAxiDma* )accelerator->dmaHardware,
                         XAXIDMA_IRQ_ALL_MASK,
                         XAXIDMA_DEVICE_TO_DMA);

    else
      XAxiDma_IntrDisable((XAxiDma* )accelerator->dmaHardware,
                          XAXIDMA_IRQ_ALL_MASK,
                          XAXIDMA_DEVICE_TO_DMA);
  }

  /***************************************************************************/
  /**************************** GIC initialization ***************************/
  if (hardware_config->mode == HARDWARE)
  {
    IntcConfig = XScuGic_LookupConfig (XPAR_SCUGIC_SINGLE_DEVICE_ID);
    ASSERT(NULL != IntcConfig);
    if (NULL == IntcConfig)
    {
      return XST_FAILURE;
    }

    status = XScuGic_CfgInitialize (&ScuGic, IntcConfig,
                                    IntcConfig->CpuBaseAddress);
    ASSERT(status == XST_SUCCESS);
    if (status != XST_SUCCESS)
    {
      return XST_FAILURE;
    }
  }

  if (hardware_config->mode == HARDWARE)
  {
    if (hardware_config->dmaRxIntVecID)
    {
      /***************************************************************************/
      /*
       * set timer0 interrupt target cpu
       */

      int intr_target_reg = XScuGic_DistReadReg(
          &ScuGic,
          XSCUGIC_SPI_TARGET_OFFSET_CALC(hardware_config->dmaRxIntVecID));

      intr_target_reg &= ~(0x000000FF << ((hardware_config->dmaRxIntVecID % 4) * 8));
      intr_target_reg |=  (0x00000001 << ((hardware_config->dmaRxIntVecID % 4) * 8)); //CPU0 ack Timer0
    //intr_target_reg |=  (0x00000002 << ((XPAR_FABRIC_AXI_TIMER_0_INTERRUPT_INTR%4)*8));//CPU1 ack Timer0

      XScuGic_DistWriteReg(
          &ScuGic,
          XSCUGIC_SPI_TARGET_OFFSET_CALC(hardware_config->dmaRxIntVecID),
          intr_target_reg);
      /***************************************************************************/
      XScuGic_SetPriorityTriggerType (&ScuGic,
                                      hardware_config->dmaRxIntVecID,
                                      0xA0, 0x3);

      status = XScuGic_Connect (&ScuGic, hardware_config->dmaRxIntVecID,
                                (Xil_InterruptHandler) Accelerator_rxInterruptHandler,
                                accelerator);
      ASSERT (status == XST_SUCCESS);
      if (status != XST_SUCCESS)
      {
        return status;
      }
      XScuGic_Enable (&ScuGic, hardware_config->dmaRxIntVecID);
    }

    if (hardware_config->dmaTxIntVecID)
    {
      /***************************************************************************/
      /*
       * set timer0 interrupt target cpu
       */

      int intr_target_reg = XScuGic_DistReadReg(
          &ScuGic,
          XSCUGIC_SPI_TARGET_OFFSET_CALC(hardware_config->dmaTxIntVecID));

      intr_target_reg &= ~(0x000000FF
          << ((hardware_config->dmaTxIntVecID % 4) * 8));
      intr_target_reg |= (0x00000001
          << ((hardware_config->dmaTxIntVecID % 4) * 8)); //CPU0 ack Timer0
    //intr_target_reg |=  (0x00000002 << ((XPAR_FABRIC_AXI_TIMER_0_INTERRUPT_INTR%4)*8));//CPU1 ack Timer0

      XScuGic_DistWriteReg(
          &ScuGic,
          XSCUGIC_SPI_TARGET_OFFSET_CALC(hardware_config->dmaTxIntVecID),
          intr_target_reg);
      /***************************************************************************/
      XScuGic_SetPriorityTriggerType (&ScuGic,
                                      hardware_config->dmaTxIntVecID,
                                      0xA0, 0x3);
      status = XScuGic_Connect (&ScuGic, hardware_config->dmaTxIntVecID,
                                (Xil_InterruptHandler) Accelerator_txInterruptHandler,
                                accelerator);
      ASSERT (status == XST_SUCCESS);
      if (status != XST_SUCCESS)
      {
        return status;
      }
      XScuGic_Enable (&ScuGic, hardware_config->dmaTxIntVecID);
    }
  }

  if (hardware_config->mode == HARDWARE)
  {
    if (hardware_config->hwIntVecID)
    {
      /***************************************************************************/
      /*
       * set timer0 interrupt target cpu
       */

      int intr_target_reg = XScuGic_DistReadReg(
          &ScuGic,
          XSCUGIC_SPI_TARGET_OFFSET_CALC(hardware_config->hwIntVecID));

      intr_target_reg &= ~(0x000000FF
          << ((hardware_config->hwIntVecID % 4) * 8));
      intr_target_reg |= (0x00000001
          << ((hardware_config->hwIntVecID % 4) * 8)); //CPU0 ack Timer0
    //intr_target_reg |=  (0x00000002 << ((XPAR_FABRIC_AXI_TIMER_0_INTERRUPT_INTR%4)*8));//CPU1 ack Timer0

      XScuGic_DistWriteReg(
          &ScuGic,
          XSCUGIC_SPI_TARGET_OFFSET_CALC(hardware_config->hwIntVecID),
          intr_target_reg);
      /***************************************************************************/
      XScuGic_SetPriorityTriggerType (&ScuGic,
                                      hardware_config->hwIntVecID,
                                      0xA0, 0x3);

      status = XScuGic_Connect (&ScuGic, hardware_config->hwIntVecID,
                                (Xil_InterruptHandler) Accelerator_hardwareInterruptHandler,
                                accelerator);
      ASSERT (status == XST_SUCCESS);
      if (status != XST_SUCCESS)
      {
        return status;
      }
      XScuGic_Enable (&ScuGic, hardware_config->hwIntVecID);
    }
  }

  /**************************** initialize ARM Core exception handlers *******/
  if (hardware_config->mode == HARDWARE)
  {
    Xil_ExceptionInit ();
    Xil_ExceptionRegisterHandler (XIL_EXCEPTION_ID_INT,
                                  (Xil_ExceptionHandler) XScuGic_InterruptHandler,
                                  (void *) &ScuGic);

    Xil_ExceptionEnable();
  }

  /***************************************************************************/
  /**************************** Accelerator initialization *******************/

  accelerator->updateHardware = hardware_config->hwDriver->new();

  ASSERT (accelerator->updateHardware != NULL);

  if (hardware_config->mode == HARDWARE)
  {
    status = hardware_config->hwDriver->Initialize (accelerator->updateHardware,
                                                    hardware_config->hwDeviceID);
    ASSERT (status == XST_SUCCESS);
    if (status != XST_SUCCESS)
    {
      xil_printf ("Sbs update hardware initialization error: %d\r\n", status);

      return XST_FAILURE;
    }


    hardware_config->hwDriver->InterruptGlobalEnable (accelerator->updateHardware);
    hardware_config->hwDriver->InterruptEnable (accelerator->updateHardware, 1);
  }
  accelerator->acceleratorReady = 1;
  accelerator->rxDone = 1;
  accelerator->txDone = 1;

  return XST_SUCCESS;
}

void Accelerator_shutdown(SbSUpdateAccelerator * accelerator)
{
  ASSERT(accelerator != NULL);
  ASSERT(accelerator->hardwareConfig != NULL);

  if ((accelerator != NULL) && (accelerator->hardwareConfig != NULL))
  {
    if (accelerator->hardwareConfig->dmaTxIntVecID)
      XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->dmaTxIntVecID);

    if (accelerator->hardwareConfig->dmaRxIntVecID)
      XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->dmaRxIntVecID);

    if (accelerator->hardwareConfig->hwIntVecID)
      XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->hwIntVecID);
  }
}

SbSUpdateAccelerator * Accelerator_new(SbSHardwareConfig * hardware_config)
{
  SbSUpdateAccelerator * accelerator = NULL;

  ASSERT (hardware_config != NULL);

  if (hardware_config != NULL)
  {
    accelerator = malloc (sizeof(SbSUpdateAccelerator));
    ASSERT (accelerator != NULL);
    if (accelerator != NULL)
    {
      int status = Accelerator_initialize(accelerator, hardware_config);
      ASSERT (status == XST_SUCCESS);

      if (status != XST_SUCCESS)
        free (accelerator);
    }
  }

  return accelerator;
}

void Accelerator_delete (SbSUpdateAccelerator ** accelerator)
{
  ASSERT(accelerator != NULL);
  ASSERT(*accelerator != NULL);

  if ((accelerator != NULL) && (*accelerator != NULL))
  {
    Accelerator_shutdown (*accelerator);
    (*accelerator)->hardwareConfig->hwDriver->delete(&(*accelerator)->updateHardware);

    free (*accelerator);
    *accelerator = NULL;
  }
}

void Accelerator_setup(SbSUpdateAccelerator * accelerator,
                              SbsAcceleratorProfie * profile,
                              AcceleratorMode mode)
{
  ASSERT (accelerator != NULL);
  ASSERT (profile != NULL);

  ASSERT (accelerator->hardwareConfig != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver != NULL);

  if (accelerator->profile != profile)
  {
    accelerator->profile = profile;

    if (accelerator->hardwareConfig->hwDriver->Set_layerSize)
      accelerator->hardwareConfig->hwDriver->Set_layerSize (
          accelerator->updateHardware, accelerator->profile->layerSize);

    if (accelerator->hardwareConfig->hwDriver->Set_kernelSize)
      accelerator->hardwareConfig->hwDriver->Set_kernelSize (
          accelerator->updateHardware, accelerator->profile->kernelSize);

    if (accelerator->hardwareConfig->hwDriver->Set_vectorSize)
      accelerator->hardwareConfig->hwDriver->Set_vectorSize (
          accelerator->updateHardware, accelerator->profile->vectorSize);

    if (accelerator->hardwareConfig->hwDriver->Set_epsilon)
      accelerator->hardwareConfig->hwDriver->Set_epsilon (
          accelerator->updateHardware, accelerator->profile->epsilon);
  }

  accelerator->mode = mode;
  if (accelerator->hardwareConfig->hwDriver->Set_mode)
    accelerator->hardwareConfig->hwDriver->Set_mode (
        accelerator->updateHardware, accelerator->mode);

  /************************** Rx Setup **************************/
  accelerator->rxBuffer = profile->rxBuffer[mode];
  accelerator->rxBufferSize = profile->rxBufferSize[mode];

  /************************** Tx Setup **************************/
  accelerator->txBuffer = profile->txBuffer[mode];
  accelerator->txBufferSize = profile->txBufferSize[mode];

  ASSERT ((uint32_t)accelerator->hardwareConfig->ddrMem.baseAddress <= (uint32_t)accelerator->rxBuffer);
  ASSERT ((uint32_t)accelerator->rxBuffer + (uint32_t)accelerator->rxBufferSize <= (uint32_t)accelerator->hardwareConfig->ddrMem.highAddress);

  ASSERT ((uint32_t)accelerator->hardwareConfig->ddrMem.baseAddress <= (uint32_t)accelerator->txBuffer);
  ASSERT ((uint32_t)accelerator->txBuffer + (uint32_t)accelerator->txBufferSize <= (uint32_t)accelerator->hardwareConfig->ddrMem.highAddress);

  accelerator->txBufferCurrentPtr = accelerator->txBuffer;

#ifdef DEBUG
  accelerator->txStateCounter = 0;
  accelerator->txWeightCounter = 0;
#endif
}

inline void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         uint32_t * state_vector) __attribute__((always_inline));

inline void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         uint32_t * state_vector)
{
  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->layerSize);
  ASSERT (0 < accelerator->profile->stateBufferSize);
  ASSERT (state_vector != NULL);

  *((uint32_t *)accelerator->txBufferCurrentPtr) = ((uint32_t) genrand ()) >> (32 - 21);

  accelerator->txBufferCurrentPtr += sizeof(uint32_t);

  memcpy(accelerator->txBufferCurrentPtr,
         state_vector,
         accelerator->profile->stateBufferSize);

  accelerator->txBufferCurrentPtr += accelerator->profile->stateBufferSize;

#ifdef DEBUG
  ASSERT(accelerator->txStateCounter <= accelerator->profile->layerSize);

  accelerator->txStateCounter ++;
#endif
}

inline void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          uint16_t * weight_vector) __attribute__((always_inline));

inline void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          uint16_t * weight_vector)
{
  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->weightBufferSize);
  ASSERT (0 < accelerator->profile->kernelSize);
  ASSERT (0 < accelerator->profile->layerSize);
  ASSERT (weight_vector != NULL);

#ifdef DEBUG
  ASSERT(accelerator->txWeightCounter <= accelerator->profile->kernelSize * accelerator->profile->layerSize);
#endif

  memcpy(accelerator->txBufferCurrentPtr,
         weight_vector,
         accelerator->profile->weightBufferSize);

  accelerator->txBufferCurrentPtr += accelerator->profile->weightBufferSize;

#ifdef DEBUG
  accelerator->txWeightCounter ++;
#endif
}

int Accelerator_start(SbSUpdateAccelerator * accelerator)
{
  int status;

  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->stateBufferSize);
  ASSERT (accelerator->mode == SPIKE_MODE || 0 < accelerator->profile->weightBufferSize);
  ASSERT (0 < accelerator->profile->layerSize);

  ASSERT ((size_t)accelerator->txBufferCurrentPtr == (size_t)accelerator->txBuffer + accelerator->txBufferSize);

#ifdef DEBUG
  ASSERT (accelerator->profile->layerSize == accelerator->txStateCounter);
#endif

  Xil_DCacheFlushRange ((UINTPTR) accelerator->txBuffer, accelerator->txBufferSize);

  while (accelerator->acceleratorReady == 0);
  while (accelerator->txDone == 0);
  while (accelerator->rxDone == 0);

  accelerator->memory_cmd = accelerator->profile->memory_cmd[accelerator->mode];

  accelerator->acceleratorReady = 0;
  accelerator->hardwareConfig->hwDriver->Start (accelerator->updateHardware);


  accelerator->txDone = 0;
  status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                         (UINTPTR) accelerator->txBuffer,
                                                         accelerator->txBufferSize,
                                                         XAXIDMA_DMA_TO_DEVICE);
  ASSERT(status == XST_SUCCESS);

  if (status == XST_SUCCESS)
  {
    accelerator->rxDone = 0;
    status = accelerator->hardwareConfig->dmaDriver->Move  (accelerator->dmaHardware,
                                                            (UINTPTR) accelerator->rxBuffer,
                                                            accelerator->rxBufferSize,
                                                            XAXIDMA_DEVICE_TO_DMA);

    ASSERT(status == XST_SUCCESS);
  }

  return status;
}

/*****************************************************************************/

Result SbsPlatform_initialize (SbSHardwareConfig * hardware_config_list,
                               uint32_t list_length)
{
  int i;
  Result rc;

  if (SbSUpdateAccelerator_list != NULL)
    free (SbSUpdateAccelerator_list);

  SbSUpdateAccelerator_list = malloc(sizeof(SbSUpdateAccelerator *) * list_length);

  ASSERT (SbSUpdateAccelerator_list != NULL);

  rc = (SbSUpdateAccelerator_list != NULL) ? OK : ERROR;

  SbSUpdateAccelerator_list_length = list_length;

  for (i = 0; (rc == OK) && (i < list_length); i++)
  {
    SbSUpdateAccelerator_list[i] = Accelerator_new (&hardware_config_list[i]);

    ASSERT (SbSUpdateAccelerator_list[i] != NULL);

    rc = SbSUpdateAccelerator_list[i] != NULL ? OK : ERROR;
  }

  return rc;
}

void SbsPlatform_shutdown (void)
{
  int i;
  ASSERT (SbSUpdateAccelerator_list != NULL);

  if (SbSUpdateAccelerator_list != NULL)
  {
    for (i = 0; i < SbSUpdateAccelerator_list_length; i++)
    {
      Accelerator_delete ((&SbSUpdateAccelerator_list[i]));
    }

    free (SbSUpdateAccelerator_list);
  }
}

/*****************************************************************************/

