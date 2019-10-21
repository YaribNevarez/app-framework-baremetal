/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: Yarib Nevarez
 */


#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "stddef.h"
#include "stdarg.h"

#include "sbs_neural_network.h"
#include "mt19937int.h"


#include "ff.h"
#include "xparameters.h"
#include "xaxidma.h"


#include "xsbs_update.h"
#include "xscugic.h"


#define ASSERT(expr)  assert(expr)

/*****************************************************************************/
#define   MEMORY_SIZE         (4771384)

#define   MAX_LAYER_SIZE      (28*28)
#define   MAX_KERNEL_SIZE     (5*5)

#define   MAX_IP_VECTOR_SIZE  (1024)

/*****************************************************************************/

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */

typedef struct
{
  size_t baseAddress;
  size_t highAddress;
  size_t blockIndex;
} MemoryBlock;

typedef struct
{
  uint32_t    updateDeviceID;
  uint32_t    dmaDeviceID;
  uint32_t    dmaTxIntVecID;
  uint32_t    dmaRxIntVecID;
  uint32_t    dmaTxBDNum;
  uint32_t    dmaRxBDNum;
  MemoryBlock ddrMem;
} SbSHardwareConfig;

typedef struct
{
  SbSHardwareConfig * hardwareConfig;
  XSbs_update         updateHardware;
  XAxiDma             dmaHardware;
  XAxiDma_BdRing *    dmaRxBdRingPtr;
  XAxiDma_BdRing *    dmaTxBdRingPtr;
  XAxiDma_Bd *        dmaFirstTxBdPtr;
  XAxiDma_Bd *        dmaFirstRxBdPtr;
  XAxiDma_Bd *        dmaCurrentTxBdPtr;
  XAxiDma_Bd *        dmaCurrentRxBdPtr;
  uint16_t            txStateCounter;
  uint16_t            txWeightCounter;

  uint16_t          vectorSize;
  uint32_t          kernelSize;
  uint32_t          layerSize;

  uint8_t           errorFlags;
} SbSUpdateAccelerator;


typedef float     Weight;
typedef uint32_t  SpikeID;

typedef struct
{
  void *   data;
  size_t   data_type_size;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
  MemoryBlock * memory_def_parent;
} Multivector;

typedef struct
{
  SbSUpdateAccelerator * accelerator;
  Multivector * state_matrix;
  Multivector * weight_matrix;
  Multivector * spike_matrix;
  Multivector * random_matrix;
} SbsPartitionLayer;

typedef enum
{
  INPUT_LAYER,
  CONVOLUTION_LAYER,
  POOLING_LAYER,
  FULLY_CONNECTED_LAYER,
  OUTPUT_LAYER
} SbsLayerType;

typedef struct
{
  SbsLayer              vtbl;
  SbsLayerType          layer_type;
  SbsPartitionLayer **  partition_array;
  uint8_t               num_partitions;
  uint16_t              kernel_size;
  uint16_t              kernel_stride;
  uint16_t              neurons_previous_Layer;
  WeightShift           weight_shift;
  float                 epsilon;
} SbsBaseLayer;

typedef struct
{
  SbsNetwork        vtbl;
  uint8_t           size;
  SbsBaseLayer **   layer_array;
  uint8_t           input_label;
  uint8_t           inferred_output;
} SbsBaseNetwork;

#pragma pack(pop)   /* restore original alignment from stack */

/*****************************************************************************/
/************************ Memory manager *************************************/


static void * MemoryBlock_alloc(MemoryBlock * memory_def, size_t size)
{
  void * ptr = NULL;

  if (memory_def != NULL
      && ((memory_def->baseAddress + memory_def->blockIndex) + size <= memory_def->highAddress))
  {
    ptr = (void *) memory_def->baseAddress + memory_def->blockIndex;
    memory_def->blockIndex += size;
  }

  return ptr;
}

/*****************************************************************************/
/************************ Accelerator ****************************************/

#if XPAR_XAXIDMA_NUM_INSTANCES != XPAR_XSBS_UPDATE_NUM_INSTANCES
  #error "DMA-SBSUpdate hardware instances mismatch"
#endif

SbSHardwareConfig HardwareConfig[] =
{
  {
    .updateDeviceID = XPAR_SBS_UPDATE_0_DEVICE_ID,
    .dmaDeviceID =XPAR_AXIDMA_0_DEVICE_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID,
    .dmaTxBDNum = MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1 + 1),
    .dmaRxBDNum = (1 + 1) * MAX_LAYER_SIZE,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x30000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x33FFFFFF,
      .blockIndex = 0
    }
  }
#if defined(XPAR_SBS_UPDATE_1_DEVICE_ID) && defined(XPAR_AXIDMA_1_DEVICE_ID)
  ,
  {
    .updateDeviceID = XPAR_SBS_UPDATE_1_DEVICE_ID,
    .dmaDeviceID =XPAR_AXIDMA_1_DEVICE_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_1_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_1_S2MM_INTROUT_VEC_ID,
    .dmaTxBDNum = MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1 + 1),
    .dmaRxBDNum = (1 + 1) * MAX_LAYER_SIZE,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x34000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x37FFFFFF,
      .blockIndex = 0
    }
  }
#endif
#if defined(XPAR_SBS_UPDATE_2_DEVICE_ID) && defined(XPAR_AXIDMA_2_DEVICE_ID)
  ,
  {
    .updateDeviceID = XPAR_SBS_UPDATE_2_DEVICE_ID,
    .dmaDeviceID =XPAR_AXIDMA_2_DEVICE_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_2_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_2_S2MM_INTROUT_VEC_ID,
    .dmaTxBDNum = MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1 + 1),
    .dmaRxBDNum = (1 + 1) * MAX_LAYER_SIZE,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x38000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3BFFFFFF,
      .blockIndex = 0
    }
  }
#endif
#if defined(XPAR_SBS_UPDATE_3_DEVICE_ID) && defined(XPAR_AXIDMA_3_DEVICE_ID)
  ,
  {
    .updateDeviceID = XPAR_SBS_UPDATE_3_DEVICE_ID,
    .dmaDeviceID =XPAR_AXIDMA_3_DEVICE_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_3_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_3_S2MM_INTROUT_VEC_ID,
    .dmaTxBDNum = MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1 + 1),
    .dmaRxBDNum = (1 + 1) * MAX_LAYER_SIZE,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3C000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3FFFFFFF,
      .blockIndex = 0
    }
  }
#endif
};

#define NUM_ACCELERATOR_INSTANCES  (sizeof(HardwareConfig) / sizeof(SbSHardwareConfig))

static SbSUpdateAccelerator ** Accelerator_array = NULL;
static XScuGic                 ScuGic = {0};

#define ACCELERATOR_DMA_RESET_TIMEOUT 10000

static void Accelerator_txInterruptHandler(void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  u32 IrqStatus = XAxiDma_BdRingGetIrq(accelerator->dmaTxBdRingPtr);

  XAxiDma_BdRingAckIrq(accelerator->dmaTxBdRingPtr, IrqStatus);

  if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
  {
    return;
  }

  if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
  {
    int TimeOut;
    XAxiDma_BdRingDumpRegs (accelerator->dmaTxBdRingPtr);

    accelerator->errorFlags |= 0x01;

    XAxiDma_Reset (&accelerator->dmaHardware);

    TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT;

    while (TimeOut)
    {
      if (XAxiDma_ResetIsDone (&accelerator->dmaHardware))
      {
        break;
      }

      TimeOut -= 1;
    }

    printf("Possible illegal address access\n");
    ASSERT(0);
    return;
  }

  if ((IrqStatus & (XAXIDMA_IRQ_DELAY_MASK | XAXIDMA_IRQ_IOC_MASK)))
  {
    XAxiDma_Bd *BdPtr;
    XAxiDma_Bd *BdCurPtr;
    int status;
    int BdCount;
    u32 BdSts;
    int Index;

    BdCount = XAxiDma_BdRingFromHw(accelerator->dmaTxBdRingPtr, XAXIDMA_ALL_BDS, &BdPtr);

    BdCurPtr = BdPtr;
    for (Index = 0; Index < BdCount; Index++)
    {
      BdSts = XAxiDma_BdGetSts(BdCurPtr);
      if ((BdSts & XAXIDMA_BD_STS_ALL_ERR_MASK) || (!(BdSts & XAXIDMA_BD_STS_COMPLETE_MASK)))
      {
        accelerator->errorFlags |= 0x02;
        break;
      }

      BdCurPtr = (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaTxBdRingPtr,
                                                   BdCurPtr);
    }

    status = XAxiDma_BdRingFree(accelerator->dmaTxBdRingPtr, BdCount, BdPtr);
    ASSERT(status == XST_SUCCESS);
  }
}

static void Accelerator_rxInterruptHandler(void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  u32 IrqStatus = XAxiDma_BdRingGetIrq(accelerator->dmaRxBdRingPtr);

  XAxiDma_BdRingAckIrq(accelerator->dmaRxBdRingPtr, IrqStatus);

  if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
  {
    return;
  }

  if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
  {
    int TimeOut;
    XAxiDma_BdRingDumpRegs (accelerator->dmaRxBdRingPtr);

    accelerator->errorFlags |= 0x01;

    XAxiDma_Reset (&accelerator->dmaHardware);

    TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT;

    while (TimeOut)
    {
      if (XAxiDma_ResetIsDone (&accelerator->dmaHardware))
      {
        break;
      }

      TimeOut -= 1;
    }

    printf("Possible illegal address access\n");
    ASSERT(0);
    return;
  }

  if ((IrqStatus & (XAXIDMA_IRQ_DELAY_MASK | XAXIDMA_IRQ_IOC_MASK)))
  {
    XAxiDma_Bd *BdPtr;
    XAxiDma_Bd *BdCurPtr;
    int status;
    int BdCount;
    u32 BdSts;
    int Index;

    BdCount = XAxiDma_BdRingFromHw(accelerator->dmaRxBdRingPtr, XAXIDMA_ALL_BDS, &BdPtr);

    BdCurPtr = BdPtr;
    for (Index = 0; Index < BdCount; Index++)
    {
      BdSts = XAxiDma_BdGetSts(BdCurPtr);
      if ((BdSts & XAXIDMA_BD_STS_ALL_ERR_MASK) || (!(BdSts & XAXIDMA_BD_STS_COMPLETE_MASK)))
      {
        accelerator->errorFlags |= 0x02;
        break;
      }

      BdCurPtr = (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaRxBdRingPtr,
                                                   BdCurPtr);
    }

    status = XAxiDma_BdRingFree(accelerator->dmaRxBdRingPtr, BdCount, BdPtr);
    ASSERT(status == XST_SUCCESS);
  }
}

static int Accelerator_initialize(SbSUpdateAccelerator * accelerator, SbSHardwareConfig * hardware_config)
{
  XScuGic_Config *    IntcConfig;
  XAxiDma_Config *    dmaConfig;
  XAxiDma_Bd          dmaBdTemplate;
  u32                 freeBdCount;
  UINTPTR             dmaRxBDBaseAddress;
  UINTPTR             dmaTxBDBaseAddress;
  int                 status;

  ASSERT (accelerator != NULL);
  ASSERT (hardware_config != NULL);

  if (accelerator == NULL || hardware_config == NULL)
    return XST_FAILURE;

  memset(accelerator, 0x00, sizeof(SbSUpdateAccelerator));

  accelerator->hardwareConfig = hardware_config;

  /******************************* DMA initialization ************************/
  dmaConfig = XAxiDma_LookupConfig (hardware_config->dmaDeviceID);
  if (dmaConfig == NULL)
  {
    xil_printf ("No configuration found for %d\r\n", hardware_config->dmaDeviceID);

    return XST_FAILURE;
  }

  status = XAxiDma_CfgInitialize (&accelerator->dmaHardware, dmaConfig);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Initialization failed %d\r\n", status);
    return XST_FAILURE;
  }

  if (!XAxiDma_HasSg(&accelerator->dmaHardware))
  {
    xil_printf ("Device configured as Simple mode \r\n");

    return XST_FAILURE;
  }

  /**************************** DMA SG RX BD initialization ******************/
  dmaRxBDBaseAddress = (UINTPTR)MemoryBlock_alloc(&hardware_config->ddrMem,
                                         XAxiDma_BdRingMemCalc(XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                                               hardware_config->dmaRxBDNum));

  if ((void *) dmaRxBDBaseAddress == NULL)
  {
    xil_printf ("Rx BD allocation error \r\n");

    return XST_FAILURE;
  }

  accelerator->dmaRxBdRingPtr = XAxiDma_GetRxRing(&accelerator->dmaHardware);

  XAxiDma_BdRingIntEnable(accelerator->dmaRxBdRingPtr, XAXIDMA_IRQ_ALL_MASK);

  XAxiDma_BdRingSetCoalesce (accelerator->dmaRxBdRingPtr, 1, 0);

  status = XAxiDma_BdRingCreate (accelerator->dmaRxBdRingPtr,
                                 dmaRxBDBaseAddress,
                                 dmaRxBDBaseAddress,
                                 XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                 hardware_config->dmaRxBDNum);

  if (status != XST_SUCCESS)
  {
    xil_printf ("RX create BD ring failed %d\r\n", status);

    return XST_FAILURE;
  }

  XAxiDma_BdClear(&dmaBdTemplate);

  status = XAxiDma_BdRingClone (accelerator->dmaRxBdRingPtr, &dmaBdTemplate);
  if (status != XST_SUCCESS)
  {
    xil_printf ("RX clone BD failed %d\r\n", status);

    return XST_FAILURE;
  }

  freeBdCount = XAxiDma_BdRingGetFreeCnt(accelerator->dmaRxBdRingPtr);

  if (freeBdCount != hardware_config->dmaRxBDNum)
  {
    xil_printf ("RX BD creation inconsistency\r\n");

    return XST_FAILURE;
  }

  status = XAxiDma_BdRingStart (accelerator->dmaRxBdRingPtr);
  if (status != XST_SUCCESS)
  {
    xil_printf ("RX start hardware failed %d\r\n", status);

    return XST_FAILURE;
  }

  /**************************** DMA SG TX BD initialization ******************/
  dmaTxBDBaseAddress = (UINTPTR)MemoryBlock_alloc(&hardware_config->ddrMem,
                                         XAxiDma_BdRingMemCalc(XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                                               hardware_config->dmaTxBDNum));

  if ((void *) dmaTxBDBaseAddress == NULL)
  {
    xil_printf ("Tx BD allocation error \r\n");

    return XST_FAILURE;
  }

  accelerator->dmaTxBdRingPtr = XAxiDma_GetTxRing(&accelerator->dmaHardware);

  XAxiDma_BdRingIntEnable(accelerator->dmaTxBdRingPtr, XAXIDMA_IRQ_ALL_MASK);

  XAxiDma_BdRingSetCoalesce(accelerator->dmaTxBdRingPtr, 1, 0);


  status = XAxiDma_BdRingCreate (accelerator->dmaTxBdRingPtr,
                                 dmaTxBDBaseAddress,
                                 dmaTxBDBaseAddress,
                                 XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                 hardware_config->dmaTxBDNum);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Failed to create BD ring in TX setup\r\n");

    return XST_FAILURE;
  }

  status = XAxiDma_BdRingClone (accelerator->dmaTxBdRingPtr, &dmaBdTemplate);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Failed to BD ring clone in TX setup %d\r\n", status);

    return XST_FAILURE;
  }

  freeBdCount = XAxiDma_BdRingGetFreeCnt(accelerator->dmaTxBdRingPtr);

  if (freeBdCount != hardware_config->dmaTxBDNum)
  {
    xil_printf ("RX BD creation inconsistency\r\n");

    return XST_FAILURE;
  }

  status = XAxiDma_BdRingStart (accelerator->dmaTxBdRingPtr);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Failed starting BD ring TX setup %d\r\n", status);

    return XST_FAILURE;
  }

  /***************************************************************************/
  /**************************** GIC initialization ***************************/
  IntcConfig = XScuGic_LookupConfig (XPAR_SCUGIC_SINGLE_DEVICE_ID);
  if (NULL == IntcConfig)
  {
    return XST_FAILURE;
  }

  status = XScuGic_CfgInitialize (&ScuGic, IntcConfig,
                                  IntcConfig->CpuBaseAddress);
  if (status != XST_SUCCESS)
  {
    return XST_FAILURE;
  }

  XScuGic_SetPriorityTriggerType (&ScuGic,
                                  hardware_config->dmaTxIntVecID,
                                  0xA0, 0x3);

  XScuGic_SetPriorityTriggerType (&ScuGic,
                                  hardware_config->dmaRxIntVecID,
                                  0xA0, 0x3);

  status = XScuGic_Connect (&ScuGic, hardware_config->dmaTxIntVecID,
                            (Xil_InterruptHandler) Accelerator_txInterruptHandler,
                            accelerator);
  if (status != XST_SUCCESS)
  {
    return status;
  }

  status = XScuGic_Connect (&ScuGic, hardware_config->dmaRxIntVecID,
                            (Xil_InterruptHandler) Accelerator_rxInterruptHandler,
                            accelerator);
  if (status != XST_SUCCESS)
  {
    return status;
  }

  XScuGic_Enable (&ScuGic, hardware_config->dmaTxIntVecID);
  XScuGic_Enable (&ScuGic, hardware_config->dmaRxIntVecID);

  /**************************** initialize ARM Core exception handlers *******/
  Xil_ExceptionInit ();
  Xil_ExceptionRegisterHandler (XIL_EXCEPTION_ID_INT,
                                (Xil_ExceptionHandler) XScuGic_InterruptHandler,
                                (void *) &ScuGic);

  Xil_ExceptionEnable();

  /***************************************************************************/
  /**************************** Accelerator initialization *******************/
  status = XSbs_update_Initialize (&accelerator->updateHardware,
                                   hardware_config->updateDeviceID);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Sbs update hardware initialization error: %d\r\n", status);

    return XST_FAILURE;
  }

  XSbs_update_InterruptGlobalDisable(&accelerator->updateHardware);

  return XST_SUCCESS;
}

static void Accelerator_shutdown(SbSUpdateAccelerator * accelerator)
{
  ASSERT(accelerator != NULL);
  ASSERT(accelerator->hardwareConfig != NULL);

  if ((accelerator != NULL) && (accelerator->hardwareConfig != NULL))
  {
    XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->dmaTxIntVecID);
    XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->dmaRxIntVecID);
  }
}

static SbSUpdateAccelerator * Accelerator_new(SbSHardwareConfig * hardware_config)
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
    free (*accelerator);
    *accelerator = NULL;
  }
}

Result SbsHardware_initialize (void)
{
  int i;
  Result rc;

  Accelerator_array = malloc (NUM_ACCELERATOR_INSTANCES * sizeof(SbSUpdateAccelerator *));
  ASSERT(Accelerator_array != NULL);

  rc = (Accelerator_array != NULL) ? OK : ERROR;

  for (i = 0; (rc == OK) && (i < NUM_ACCELERATOR_INSTANCES); i++)
  {
    Accelerator_array[i] = Accelerator_new (&HardwareConfig[i]);

    ASSERT (Accelerator_array[i] != NULL);

    rc = Accelerator_array[i] != NULL ? OK : ERROR;
  }

  return rc;
}

void SbsHardware_shutdown (void)
{
  int i;
  ASSERT (Accelerator_array != NULL);

  if (Accelerator_array != NULL)
  {
    for (i = 0; i < sizeof(HardwareConfig) / sizeof(SbSHardwareConfig); i++)
    {
      Accelerator_delete ((&Accelerator_array[i]));
    }

    free (Accelerator_array);
  }
}

static void Accelerator_setup(SbSUpdateAccelerator * accelerator,
                              uint32_t      layerSize,
                              uint32_t      kernelSize,
                              uint16_t      vectorSize,
                              float         epsilon)
{
  int status;
  ASSERT (accelerator != NULL);
  ASSERT (0 < layerSize);
  ASSERT (0 < kernelSize);
  ASSERT (0 < vectorSize);
  ASSERT (0.0 < epsilon);

  XSbs_update_Set_layerSize (&accelerator->updateHardware, layerSize);
  accelerator->layerSize = layerSize;

  XSbs_update_Set_kernelSize (&accelerator->updateHardware, kernelSize);
  accelerator->kernelSize = kernelSize;

  XSbs_update_Set_vectorSize (&accelerator->updateHardware, vectorSize);
  accelerator->vectorSize = vectorSize;

  XSbs_update_Set_epsilon (&accelerator->updateHardware, *(uint32_t*) &epsilon);

  while (accelerator->dmaTxBdRingPtr->PostCnt);
  while (accelerator->dmaRxBdRingPtr->PostCnt);

  ASSERT(0 < XAxiDma_BdRingGetFreeCnt(accelerator->dmaTxBdRingPtr));

  status = XAxiDma_BdRingAlloc (accelerator->dmaTxBdRingPtr,
                                accelerator->layerSize * (accelerator->kernelSize + 1 + 1),
                                &accelerator->dmaFirstTxBdPtr);
  ASSERT (status == XST_SUCCESS);

  XAxiDma_BdSetCtrl (accelerator->dmaFirstTxBdPtr, XAXIDMA_BD_CTRL_TXSOF_MASK);

  accelerator->dmaCurrentTxBdPtr = accelerator->dmaFirstTxBdPtr;

  /************************** Rx Setup **************************/
  ASSERT(0 < XAxiDma_BdRingGetFreeCnt(accelerator->dmaRxBdRingPtr));

  status = XAxiDma_BdRingAlloc (accelerator->dmaRxBdRingPtr,
                                2 * layerSize,
                                &accelerator->dmaFirstRxBdPtr);
  ASSERT(status == XST_SUCCESS);

  accelerator->dmaCurrentRxBdPtr = accelerator->dmaFirstRxBdPtr;

  accelerator->txStateCounter = 0;
  accelerator->txWeightCounter = 0;
}

static void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         NeuronState * state_vector,
                                         SpikeID * spike_id,
                                         uint32_t * random_number)
{
  int status;

  ASSERT (accelerator != NULL);
  ASSERT (state_vector != NULL);
  ASSERT (0 < accelerator->vectorSize);
  ASSERT (spike_id != NULL);
  ASSERT (random_number != NULL);

  /************************** Tx random_number *******************************/
  NeuronState random_s = ((NeuronState) genrand ()) * (1.0/((NeuronState) 0xFFFFFFFF));
  *(float*)random_number = random_s;

  Xil_DCacheFlushRange ((UINTPTR) random_number, sizeof(uint32_t));

  ASSERT (accelerator->dmaCurrentTxBdPtr != NULL);
  status = XAxiDma_BdSetBufAddr (accelerator->dmaCurrentTxBdPtr, (UINTPTR) random_number);
  ASSERT (status == XST_SUCCESS);

  status = XAxiDma_BdSetLength (accelerator->dmaCurrentTxBdPtr,
                                sizeof(uint32_t),
                                accelerator->dmaTxBdRingPtr->MaxTransferLen);
  ASSERT (status == XST_SUCCESS);

  accelerator->dmaCurrentTxBdPtr =
      (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaTxBdRingPtr,
                                        accelerator->dmaCurrentTxBdPtr);

  ASSERT (accelerator->dmaCurrentTxBdPtr != NULL);

  /************************** Tx state_vector ********************************/
  Xil_DCacheFlushRange ((UINTPTR) state_vector, accelerator->vectorSize * sizeof(NeuronState));

  ASSERT (accelerator->dmaCurrentTxBdPtr != NULL);
  status = XAxiDma_BdSetBufAddr (accelerator->dmaCurrentTxBdPtr, (UINTPTR) state_vector);
  ASSERT (status == XST_SUCCESS);

  status = XAxiDma_BdSetLength (accelerator->dmaCurrentTxBdPtr,
                                accelerator->vectorSize * sizeof(NeuronState),
                                accelerator->dmaTxBdRingPtr->MaxTransferLen);
  ASSERT (status == XST_SUCCESS);

  accelerator->dmaCurrentTxBdPtr =
      (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaTxBdRingPtr,
                                        accelerator->dmaCurrentTxBdPtr);

  ASSERT (accelerator->dmaCurrentTxBdPtr != NULL);

  /************************** Rx spike_id ************************************/
  ASSERT (accelerator->dmaCurrentRxBdPtr != NULL);
  status = XAxiDma_BdSetBufAddr (accelerator->dmaCurrentRxBdPtr,
                                 (UINTPTR) spike_id);
  ASSERT(status == XST_SUCCESS);

  ASSERT(0 < accelerator->vectorSize);
  status = XAxiDma_BdSetLength (accelerator->dmaCurrentRxBdPtr,
                                sizeof(SpikeID),
                                accelerator->dmaRxBdRingPtr->MaxTransferLen);
  ASSERT(status == XST_SUCCESS);

  accelerator->dmaCurrentRxBdPtr =
      (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaRxBdRingPtr,
                                        accelerator->dmaCurrentRxBdPtr);

  ASSERT (accelerator->dmaCurrentRxBdPtr != NULL);

  /************************** Rx state_vector ********************************/
  ASSERT (accelerator->dmaCurrentRxBdPtr != NULL);
  status = XAxiDma_BdSetBufAddr (accelerator->dmaCurrentRxBdPtr,
                                 (UINTPTR) state_vector);
  ASSERT(status == XST_SUCCESS);

  ASSERT(0 < accelerator->vectorSize);
  status = XAxiDma_BdSetLength (accelerator->dmaCurrentRxBdPtr,
                                accelerator->vectorSize * sizeof(NeuronState),
                                accelerator->dmaRxBdRingPtr->MaxTransferLen);
  ASSERT(status == XST_SUCCESS);

  accelerator->dmaCurrentRxBdPtr =
      (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaRxBdRingPtr,
                                        accelerator->dmaCurrentRxBdPtr);

  ASSERT (accelerator->dmaCurrentRxBdPtr != NULL);

  accelerator->txStateCounter ++;
  ASSERT(accelerator->txStateCounter <= accelerator->layerSize);
}

static void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          Weight * weight_vector)
{
  int status;

  ASSERT (weight_vector != NULL);

  Xil_DCacheFlushRange ((UINTPTR) weight_vector, accelerator->vectorSize * sizeof(Weight));

  ASSERT (accelerator->dmaCurrentTxBdPtr != NULL);
  ASSERT (accelerator->dmaCurrentTxBdPtr != accelerator->dmaFirstTxBdPtr);

  /* Set up the BD using the information of the packet to transmit */
  status = XAxiDma_BdSetBufAddr (accelerator->dmaCurrentTxBdPtr, (UINTPTR) weight_vector);
  ASSERT(status == XST_SUCCESS);

  status = XAxiDma_BdSetLength (accelerator->dmaCurrentTxBdPtr,
                                accelerator->vectorSize * sizeof(Weight),
                                accelerator->dmaTxBdRingPtr->MaxTransferLen);
  ASSERT(status == XST_SUCCESS);

  accelerator->txWeightCounter ++;

  ASSERT(accelerator->txWeightCounter <= accelerator->kernelSize * accelerator->layerSize);

  if (accelerator->txWeightCounter < accelerator->kernelSize * accelerator->layerSize)
  {
    accelerator->dmaCurrentTxBdPtr =
        (XAxiDma_Bd *) XAxiDma_BdRingNext(accelerator->dmaTxBdRingPtr,
                                          accelerator->dmaCurrentTxBdPtr);

    ASSERT(accelerator->dmaCurrentTxBdPtr != NULL);
  }
  else
  {
    ASSERT(accelerator->txWeightCounter == accelerator->kernelSize * accelerator->layerSize);
    XAxiDma_BdSetCtrl (accelerator->dmaCurrentTxBdPtr, XAXIDMA_BD_CTRL_TXEOF_MASK);
  }
}

static void Accelerator_start(SbSUpdateAccelerator * accelerator)
{
  uint32_t allocatedDmaBd;
  int status;

  while (!XSbs_update_IsReady (&accelerator->updateHardware));

  XSbs_update_Start (&accelerator->updateHardware);

  allocatedDmaBd = accelerator->layerSize * (accelerator->kernelSize + 1 + 1);

  ASSERT (allocatedDmaBd == accelerator->dmaTxBdRingPtr->PreCnt);
  ASSERT (accelerator->layerSize == accelerator->txStateCounter);

  status = XAxiDma_BdRingToHw (accelerator->dmaTxBdRingPtr,
                               allocatedDmaBd,
                               accelerator->dmaFirstTxBdPtr);
  ASSERT(status == XST_SUCCESS);


  ASSERT (2 * accelerator->layerSize == accelerator->dmaRxBdRingPtr->PreCnt);

  status = XAxiDma_BdRingToHw (accelerator->dmaRxBdRingPtr,
                               2 * accelerator->layerSize,
                               accelerator->dmaFirstRxBdPtr);
  ASSERT(status == XST_SUCCESS);

  accelerator->dmaFirstTxBdPtr = NULL;
  accelerator->dmaFirstRxBdPtr = NULL;
  accelerator->dmaCurrentTxBdPtr = NULL;
  accelerator->dmaCurrentRxBdPtr = NULL;
}

/*****************************************************************************/
/*****************************************************************************/

static Multivector * Multivector_new(MemoryBlock * memory_def, uint8_t data_type_size, uint8_t dimensionality, ...)
{
  Multivector * multivector = NULL;

  ASSERT(0 <= dimensionality);

  if (0 <= dimensionality)
  {
    size_t memory_size = sizeof(Multivector) + (dimensionality - 1) * sizeof(uint16_t);
    multivector = malloc (memory_size);

    ASSERT(multivector != NULL);

    if (multivector != NULL)
    {
      int arg;
      size_t data_size;
      va_list argument_list;

      memset (multivector, 0x00, memory_size);

      va_start(argument_list, dimensionality);

      for (data_size = 1, arg = 0; arg < dimensionality; arg ++)
        data_size *= (multivector->dimension_size[arg] = (uint16_t) va_arg(argument_list, int));

      va_end(argument_list);

      multivector->memory_def_parent = memory_def;

      if (memory_def != NULL)
        multivector->data = MemoryBlock_alloc (memory_def,
                                               data_size * data_type_size);
      else
        multivector->data = malloc (data_size * data_type_size);

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size * data_type_size);

      multivector->dimensionality = dimensionality;
      multivector->data_type_size = data_type_size;
    }
  }

  return multivector;
}

void * Multivector_2DAccess(Multivector * multivector, uint16_t row, uint16_t column)
{
  void * data = NULL;
  ASSERT (multivector != NULL);
  ASSERT (multivector->data != NULL);
  ASSERT (2 <= multivector->dimensionality);
  ASSERT (row <= multivector->dimension_size[0]);
  ASSERT (column <= multivector->dimension_size[1]);

  if ((multivector != NULL)
      && (multivector->data != NULL)
      && (2 <= multivector->dimensionality)
      && (row <= multivector->dimension_size[0])
      && (column <= multivector->dimension_size[1]))
  {
    uint16_t dimensionality = multivector->dimensionality;
    size_t data_size = multivector->data_type_size;

    while (dimensionality-- > 2)
    {
      data_size *= multivector->dimension_size[dimensionality];
    }

    data = multivector->data
        + (row * multivector->dimension_size[1] + column) * data_size;
  }

  return data;
}

static Multivector * Multivector_duplicate(MemoryBlock * memory_def,
                                           Multivector * original)
{
  Multivector * duplicate = NULL;
  ASSERT(original != NULL);
  ASSERT(0 < original->dimensionality);

  if ((original != NULL)
      && (0 < original->dimensionality))
  {
    size_t memory_size = sizeof(Multivector)
        + (original->dimensionality - 1) * sizeof(uint16_t);
    duplicate = malloc (memory_size);

    ASSERT(duplicate != NULL);

    if (duplicate != NULL)
    {
      size_t data_size = original->data_type_size;
      int i;

      memcpy (duplicate, original, memory_size);

      for (i = 0; i < original->dimensionality; i++)
        data_size *= original->dimension_size[i];

      duplicate->memory_def_parent = memory_def;

      if (memory_def != NULL)
        duplicate->data = MemoryBlock_alloc (memory_def, data_size);
      else
        duplicate->data = malloc (data_size);

      ASSERT(duplicate->data != NULL);

      if (duplicate->data != NULL)
        memcpy (duplicate->data, original->data, data_size);
      else
        return NULL;
    }
  }
  return duplicate;
}

static void Multivector_delete(Multivector ** multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(*multivector != NULL);

  if ((multivector != NULL) && (*multivector != NULL))
  {
    if ((*multivector)->memory_def_parent == NULL)
      free ((*multivector)->data);

    free(*multivector);
    *multivector = NULL;
  }
}

/*****************************************************************************/
static SbsPartitionLayer * SbsPartitionLayer_new (SbSUpdateAccelerator * accelerator,
                                                  uint16_t rows,
                                                  uint16_t columns,
                                                  uint16_t neurons)
{
  SbsPartitionLayer * partition = NULL;

  partition = (SbsPartitionLayer *) malloc (sizeof(SbsPartitionLayer));

  ASSERT (partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;
    Multivector * random_matrix = NULL;
    MemoryBlock * memory_def = NULL;

    memset (partition, 0x00, sizeof(SbsPartitionLayer));

    if (accelerator != NULL)
    {
      ASSERT (accelerator->hardwareConfig != NULL);

      if (accelerator->hardwareConfig != NULL)
      {
        partition->accelerator = accelerator;
        memory_def = &accelerator->hardwareConfig->ddrMem;
      }
    }

    /* Instantiate state_matrix */
    state_matrix = Multivector_new (memory_def, sizeof(NeuronState), 3, rows,
                                    columns, neurons);

    ASSERT(state_matrix != NULL);
    ASSERT(state_matrix->dimensionality == 3);
    ASSERT(state_matrix->data != NULL);
    ASSERT(state_matrix->dimension_size[0] == rows);
    ASSERT(state_matrix->dimension_size[1] == columns);
    ASSERT(state_matrix->dimension_size[2] == neurons);

    partition->state_matrix = state_matrix;

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new (memory_def, sizeof(SpikeID), 2, rows,
                                    columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    partition->spike_matrix = spike_matrix;

    /* Instantiate random_matrix */
    random_matrix = Multivector_new (memory_def, sizeof(uint32_t), 2, rows,
                                     columns);

    ASSERT(random_matrix != NULL);
    ASSERT(random_matrix->dimensionality == 2);
    ASSERT(random_matrix->data != NULL);
    ASSERT(random_matrix->dimension_size[0] == rows);
    ASSERT(random_matrix->dimension_size[1] == columns);

    partition->random_matrix = random_matrix;
  }

  return partition;
}

static void SbsPartitionLayer_delete(SbsPartitionLayer ** partition)
{
  ASSERT (partition != NULL);
  ASSERT (*partition != NULL);

  if ((partition != NULL) && (*partition != NULL))
  {
    Multivector_delete (&((*partition)->state_matrix));
    Multivector_delete (&((*partition)->spike_matrix));
    Multivector_delete (&((*partition)->random_matrix));
    Multivector_delete (&((*partition)->weight_matrix));

    free (*partition);

    *partition = NULL;
  }
}

static void SbsPartitionLayer_initializeIP(NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
      float    initial_value_h = 1.0f / size;
      uint16_t neuron;
      for (neuron = 0; neuron < size; neuron ++)
        state_vector[neuron] = initial_value_h;
  }
}

static void SbsPartitionLayer_initialize (SbsPartitionLayer * partition)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = partition->state_matrix;
    uint16_t rows = state_matrix->dimension_size[0];
    uint16_t columns = state_matrix->dimension_size[1];
    uint16_t neurons = state_matrix->dimension_size[2];
    NeuronState * state_matrix_data = state_matrix->data;

    uint16_t row;
    uint16_t column;
    size_t current_row_index;

    for (row = 0; row < rows; row++)
    {
      current_row_index = row * columns * neurons;
      for (column = 0; column < columns; column++)
      {
        SbsPartitionLayer_initializeIP (&state_matrix_data[current_row_index + column * neurons], neurons);
      }
    }
  }
}

static void SbsPartitionLayer_setWeights (SbsPartitionLayer * partition,
                                           SbsWeightMatrix weight_matrix)
{
  ASSERT(partition != NULL);
  ASSERT(partition->accelerator != NULL);
  ASSERT(partition->accelerator->hardwareConfig != NULL);
  ASSERT(weight_matrix != NULL);

  if ((partition != NULL)
      && (partition->accelerator != NULL)
      && (partition->accelerator->hardwareConfig != NULL)
      && (weight_matrix != NULL))
  {
    if (partition->weight_matrix != NULL)
      Multivector_delete(&partition->weight_matrix);

    partition->weight_matrix =
        Multivector_duplicate(&partition->accelerator->hardwareConfig->ddrMem,
                              weight_matrix);
  }
}

/*****************************************************************************/

static SbsLayer * SbsBaseLayer_new(SbsLayerType layer_type,
                                   uint16_t rows,
                                   uint16_t columns,
                                   uint16_t neurons,
                                   uint16_t kernel_size,
                                   uint16_t kernel_stride,
                                   WeightShift weight_shift,
                                   uint16_t    neurons_previous_Layer)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);

  if (layer != NULL)
  {
    int i;
    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    layer->layer_type = layer_type;

    switch (layer_type)
    {
      case CONVOLUTION_LAYER:
      case POOLING_LAYER:
        layer->num_partitions = NUM_ACCELERATOR_INSTANCES;
        break;
      case INPUT_LAYER:
      case FULLY_CONNECTED_LAYER:
      case OUTPUT_LAYER:
        layer->num_partitions = 1;
        break;
      default:
        ASSERT (0);
    }

    layer->partition_array = (SbsPartitionLayer **)
        malloc (layer->num_partitions * sizeof(SbsPartitionLayer *));
    ASSERT(layer->partition_array != NULL);

    if (layer->partition_array != NULL)
    {
      SbSUpdateAccelerator * accelerator = NULL;
      for (i = 0; i < layer->num_partitions; i++)
      {
        if (layer_type != INPUT_LAYER)
          accelerator = Accelerator_array[i];

        layer->partition_array[i] = SbsPartitionLayer_new (accelerator, rows,
                                                           columns, neurons);
      }
    }

    /* Assign parameters */
    layer->kernel_size   = kernel_size;
    layer->kernel_stride = kernel_stride;
    layer->weight_shift  = weight_shift;
    layer->neurons_previous_Layer = neurons_previous_Layer;
  }

  return (SbsLayer *) layer;
}

static void SbsBaseLayer_delete(SbsLayer ** layer_ptr)
{
  ASSERT(layer_ptr!= NULL);
  ASSERT(*layer_ptr!= NULL);
  if ((layer_ptr != NULL) && (*layer_ptr != NULL))
  {
    SbsBaseLayer ** layer = (SbsBaseLayer **) layer_ptr;

    if ((*layer)->partition_array != NULL)
      while ((*layer)->num_partitions)
        SbsPartitionLayer_delete (&((*layer)->partition_array[--(*layer)->num_partitions]));

    free (*layer);
    *layer = NULL;
  }
}

static void SbsBaseLayer_initialize(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    int i;
    for (i = 0; i < layer->num_partitions; i++)
      SbsPartitionLayer_initialize (layer->partition_array[i]);
  }
}

static void SbsBaseLayer_setWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if ((layer != NULL) && (((SbsBaseLayer *) layer)->partition_array != NULL))
  {
    int i;

    for (i = 0; i < ((SbsBaseLayer *) layer)->num_partitions; i++)
      SbsPartitionLayer_setWeights (((SbsBaseLayer *) layer)->partition_array[i],
                                     weight_matrix);
  }
}

static void SbsBaseLayer_setEpsilon(SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->epsilon = epsilon;
}

static SpikeID SbsStateVector_generateSpikeIP (NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    NeuronState random_s = ((NeuronState) genrand ()) * (1.0/((NeuronState) 0xFFFFFFFF));
    NeuronState sum      = 0.0f;
    SpikeID     spikeID;

    ASSERT(random_s <= 1.0F);

    for (spikeID = 0; spikeID < size; spikeID++)
    {
      sum += state_vector[spikeID];

      ASSERT(sum <= 1 + 1e-5);

      if (random_s <= sum)
        return spikeID;
    }
  }

  return size - 1;
}

static void SbsPartitionLayer_generateSpikes(SbsPartitionLayer * partition)
{
  ASSERT(partition != NULL);
  if (partition != NULL)
  {
    Multivector * state_matrix = partition->state_matrix;
    uint16_t rows = state_matrix->dimension_size[0];
    uint16_t columns = state_matrix->dimension_size[1];
    uint16_t neurons = state_matrix->dimension_size[2];
    NeuronState * state_matrix_data = state_matrix->data;
    SpikeID * spike_matrix_data = partition->spike_matrix->data;

    uint16_t row;
    uint16_t column;
    size_t current_row_index;
    size_t current_row_column_index;

    for (row = 0; row < rows; row++)
    {
      current_row_index = columns * row;
      for (column = 0; column < columns; column++)
      {
        current_row_column_index = current_row_index + column;
        spike_matrix_data[current_row_column_index] =
            SbsStateVector_generateSpikeIP (&state_matrix_data[current_row_column_index * neurons],
                                            neurons);
      }
    }
  }
}

static void SbsBaseLayer_generateSpikes(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);

  if ((layer != NULL) && (layer->partition_array != NULL))
  {
    int i;
    for (i = 0; i < ((SbsBaseLayer *) layer)->num_partitions; i++)
      SbsPartitionLayer_generateSpikes (((SbsBaseLayer *) layer)->partition_array[i]);
  }
}

static void SbsBaseLayer_update(SbsBaseLayer * layer, Multivector * input_spike_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(layer->state_matrix != NULL);
  ASSERT(layer->state_matrix->data != NULL);
  ASSERT(layer->weight_matrix != NULL);
  ASSERT(layer->weight_matrix->data != NULL);

  ASSERT(0 < layer->kernel_size);

  ASSERT(input_spike_matrix != NULL);
  ASSERT(input_spike_matrix->data != NULL);

  if (   (layer != NULL)
      && (layer->state_matrix != NULL)
      && (layer->state_matrix->data != NULL)
      && (layer->weight_matrix != NULL)
      && (layer->weight_matrix->data != NULL)
      && (input_spike_matrix != NULL)
      && (input_spike_matrix->data != NULL))
  {
    SpikeID   spikeID       = 0;
    SpikeID * spike_data    = input_spike_matrix->data;
    uint16_t  spike_rows    = input_spike_matrix->dimension_size[0];
    uint16_t  spike_columns = input_spike_matrix->dimension_size[1];

    NeuronState * weight_data    = layer->weight_matrix->data;
    NeuronState * weight_vector  = NULL;
    uint16_t      weight_columns = layer->weight_matrix->dimension_size[1];

    Multivector * state_matrix   = layer->state_matrix;
    NeuronState * state_data     = state_matrix->data;
    NeuronState * state_vector   = NULL;
    uint16_t      state_row_size = state_matrix->dimension_size[1] * state_matrix->dimension_size[2];
    uint16_t      neurons        = state_matrix->dimension_size[2];

    uint16_t kernel_stride  = layer->kernel_stride;
    uint16_t kernel_size    = layer->kernel_size;
    uint16_t row_shift      = kernel_size;
    uint16_t column_shift   = 1;
    uint16_t section_shift  = 0;


    uint16_t layer_row;         /* Row index for navigation on the layer */
    uint16_t layer_column;      /* Column index for navigation on the layer */
    uint16_t kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
    uint16_t kernel_row_pos;    /* Kernel row position for navigation on the spike matrix */
    uint16_t kernel_row;        /* Row index for navigation inside kernel */
    uint16_t kernel_column;     /* Column index for navigation inside kernel */

    uint16_t  spike_row_index;

    uint16_t neurons_previous_Layer = layer->neurons_previous_Layer;

    float epsilon = layer->epsilon;

    uint32_t *  random_matrix = (uint32_t *) layer->random_matrix->data;
    SpikeID *   spike_matrix = (SpikeID *) layer->spike_matrix->data;
    uint16_t    spike_row_size = layer->spike_matrix->dimension_size[1];

    ASSERT(weight_columns == neurons);

    if (weight_columns != neurons)
      return;

    if (layer->weight_shift == ROW_SHIFT)
    {
      row_shift = 1;
      column_shift = kernel_size;
    }

    Accelerator_setup (&Accelerator[0],
                       state_matrix->dimension_size[0] * state_matrix->dimension_size[1],
                       kernel_size * kernel_size,
                       neurons,
                       epsilon);


    /* Update begins */
    for (kernel_row_pos = 0, layer_row = 0;
         kernel_row_pos < spike_rows - (kernel_size - 1);
         kernel_row_pos += kernel_stride, layer_row ++)
    {
      for (kernel_column_pos = 0, layer_column = 0;
           kernel_column_pos < spike_columns - (kernel_size - 1);
           kernel_column_pos += kernel_stride, layer_column ++)
      {
        state_vector = &state_data[layer_row * state_row_size + layer_column * neurons];
        //state_vector = Multivector_2DAccess(layer->state_matrix, layer_row, layer_column);

        Accelerator_giveStateVector (&Accelerator[0],
                                     state_vector,
                                     &spike_matrix[layer_row * spike_row_size + layer_column],
                                     &random_matrix[layer_row * spike_row_size + layer_column]);

        for (kernel_row = 0; kernel_row < kernel_size; kernel_row ++)
        {
            spike_row_index = (kernel_row_pos + kernel_row) * spike_columns;
          for (kernel_column = 0; kernel_column < kernel_size; kernel_column ++)
          {
            spikeID = spike_data[spike_row_index + kernel_column_pos + kernel_column];
            //spikeID = *(SpikeID*)Multivector_2DAccess(input_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

            section_shift = (kernel_row * row_shift + kernel_column * column_shift) * neurons_previous_Layer;

            weight_vector = &weight_data[(spikeID + section_shift) * weight_columns];

            Accelerator_giveWeightVector (&Accelerator[0],
                                          weight_vector);
            if (0)
            SbsBaseLayer_updateIP (layer, state_vector, weight_vector, neurons, epsilon);

          }
        }
      }
    }

        Accelerator_start(&Accelerator[0]);

    /* Update ends*/
  }
}

/*****************************************************************************/

static SbsNetwork * SbsBaseNetwork_new(void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc (sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
    memset (network, 0x0, sizeof(SbsBaseNetwork));
    network->vtbl = _SbsNetwork;
    network->input_label = (uint8_t) -1;
    network->inferred_output = (uint8_t) -1;

    sgenrand (666); /*TODO: Create MT19937 object wrapper */
  }

  ASSERT(network->size == 0);
  ASSERT(network->layer_array == NULL);

  return (SbsNetwork *) network;
}

static void SbsBaseNetwork_delete(SbsNetwork ** network_ptr)
{
  ASSERT(network_ptr != NULL);
  ASSERT(*network_ptr != NULL);

  if ((network_ptr != NULL) && (*network_ptr != NULL))
  {
    SbsBaseNetwork ** network = (SbsBaseNetwork **) network_ptr;
    while (0 < (*network)->size)
      SbsBaseLayer_delete((SbsLayer **)&(*network)->layer_array[--((*network)->size)]);

    free(*network);
    *network = NULL;
  }
}

static void SbsBaseNetwork_giveLayer(SbsNetwork * network_ptr, SbsLayer * layer)
{
  ASSERT(network_ptr != NULL);
  ASSERT(layer != NULL);

  if ((network_ptr != NULL) && (layer != NULL))
  {
    SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
    SbsBaseLayer ** layer_array = network->layer_array;
    uint8_t size = network->size;

    ASSERT(size < 0xFF);

    layer_array = realloc(layer_array, (size + 1) * sizeof(SbsBaseLayer *));

    ASSERT(layer_array != NULL);

    if (layer_array != NULL)
    {
        layer_array[size] = (SbsBaseLayer *)layer;

        network->layer_array = layer_array;
        network->size ++;
    }
  }
}

static void SbsBaseNetwork_loadInput(SbsNetwork * network_ptr, char * file_name)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(1 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(*network->layer_array != NULL);

  ASSERT(file_name != NULL);

  if ((network != NULL)
      && (1 <= network->size)
      && (network->layer_array != NULL) && (*network->layer_array != NULL)
      && (file_name != NULL))
  {
#ifdef USE_XILINX
    FIL fil; /* File object */
    FRESULT rc;
    rc = f_open (&fil, file_name, FA_READ);
    ASSERT(rc == FR_OK);

    if (rc == FR_OK)
    {
      SbsBaseLayer * input_layer = network->layer_array[0];
      uint16_t       rows        = input_layer->state_matrix->dimension_size[0];
      uint16_t       columns     = input_layer->state_matrix->dimension_size[1];
      uint16_t       neurons     = input_layer->state_matrix->dimension_size[2];
      NeuronState *  data        = input_layer->state_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t   read_result = 0;

      uint8_t good_reading_flag = 1;

      size_t inference_population_size = sizeof(NeuronState) * neurons;

      for (column = 0; (column < columns) && good_reading_flag; column++)
        for (row = 0; (row < rows) && good_reading_flag; row++)
        {
          rc = f_read (&fil, &data[column * neurons + row * columns * neurons],
                       inference_population_size, &read_result);

          good_reading_flag = read_result == inference_population_size;
        }

      if (good_reading_flag)
      {
        rc = f_read (&fil, &network->input_label, sizeof(uint8_t), &read_result);
        network->input_label--;
        good_reading_flag = read_result == sizeof(uint8_t);
      }

      f_close (&fil);
      ASSERT(good_reading_flag);
    }
#else
    FILE * file = fopen(file_name, "rb");

    ASSERT(file != NULL);

    if (file != NULL)
    {
      SbsBaseLayer * input_layer = network->layer_array[0];
      uint16_t rows = input_layer->state_matrix->dimension_size[0];
      uint16_t columns = input_layer->state_matrix->dimension_size[1];
      uint16_t neurons = input_layer->state_matrix->dimension_size[2];
      NeuronState * data = input_layer->state_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t read_result = 0;

      uint8_t good_reading_flag = 1;

      size_t inference_population_size = sizeof(NeuronState) * neurons;

      for (column = 0; (column < columns) && good_reading_flag; column++)
        for (row = 0; (row < rows) && good_reading_flag; row++)
        {
          read_result = fread (&data[column * neurons + row * columns * neurons], 1,
              inference_population_size, file);

          good_reading_flag = read_result == inference_population_size;
        }

      if (good_reading_flag)
      {
        read_result = fread(&network->input_label, 1, sizeof(uint8_t), file);
        network->input_label --;
        good_reading_flag = read_result == sizeof(uint8_t);
      }

      fclose(file);
      ASSERT(good_reading_flag);
    }
#endif
  }
}

static void SbsBaseNetwork_updateCycle(SbsNetwork * network_ptr, uint16_t cycles)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    uint16_t i;
    /* Initialize all layers except the input-layer */
    for (i = 1; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
    }

    /************************ Begins Update cycle **************************/
    while (cycles--)
    {
      for (i = 0; i < network->size; i++)
      {

        if (i == 0) //if (i < network->size - 1)
          SbsBaseLayer_generateSpikes(network->layer_array[i]);

        if (0 < i)
          SbsBaseLayer_update(network->layer_array[i],
              network->layer_array[i - 1]->spike_matrix);
      }

      if (cycles % 100 == 0)
        printf("%d\n", cycles);
    }
    /************************ Ends Update cycle ****************************/

    /************************ Get inferred output **************************/
    {
      NeuronState max_value = 0;
      SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
      Multivector * output_state_matrix = output_layer->state_matrix;
      NeuronState * output_state_vector = output_state_matrix->data;

      ASSERT(output_state_matrix->dimensionality == 3);
      ASSERT(output_state_matrix->dimension_size[0] == 1);
      ASSERT(output_state_matrix->dimension_size[1] == 1);
      ASSERT(0 < output_state_matrix->dimension_size[2]);

      for (i = 0; i < output_state_matrix->dimension_size[2]; i++)
      {
        NeuronState h = output_state_vector[i]; /* Ensure data alignment */
        if (max_value < h)
        {
          network->inferred_output = i;
          max_value = h;
        }
      }
    }
  }
}

static uint8_t SbsBaseNetwork_getInferredOutput(SbsNetwork * network)
{
  uint8_t inferred_output = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    inferred_output = ((SbsBaseNetwork *) network)->inferred_output;
  }

  return inferred_output;
}

static uint8_t SbsBaseNetwork_getInputLabel(SbsNetwork * network)
{
  uint8_t input_label = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    input_label = ((SbsBaseNetwork *) network)->input_label;
  }

  return input_label;
}

static void SbsBaseNetwork_getOutputVector(SbsNetwork * network_ptr, NeuronState ** output_vector, uint16_t * output_vector_size)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(0 < network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(network->layer_array[network->size - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != NULL);

  if ((network != NULL)
      && (0 < network->size)
      && (network->layer_array != NULL)
      && (network->layer_array != NULL)
      && (output_vector != NULL)
      && (output_vector_size != NULL))
  {
    SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
    Multivector * output_state_matrix = output_layer->state_matrix;

    ASSERT(output_state_matrix->data != NULL);
    ASSERT(output_state_matrix->dimensionality == 3);
    ASSERT(output_state_matrix->dimension_size[0] == 1);
    ASSERT(output_state_matrix->dimension_size[1] == 1);
    ASSERT(0 < output_state_matrix->dimension_size[2]);

    * output_vector = output_state_matrix->data;
    * output_vector_size = output_state_matrix->dimension_size[2];
  }
}

static size_t SbsBaseNetwork_getMemorySize(SbsNetwork * network)
{
  return 0; //Memory_getBlockSize();
}
/*****************************************************************************/

static SbsLayer * SbsInputLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new (INPUT_LAYER, rows, columns, neurons, 0,
                                        0, ROW_SHIFT, 0);
}

static SbsLayer * SbsConvolutionLayer_new(uint16_t rows,
                                            uint16_t columns,
                                            uint16_t neurons,
                                            uint16_t kernel_size,
                                            WeightShift weight_shift,
                                            uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (CONVOLUTION_LAYER, rows, columns,
                                        neurons, kernel_size, 1, weight_shift,
                                        neurons_prev_Layer);
}

static SbsLayer * SbsPoolingLayer_new(uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift,
                                    uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (POOLING_LAYER, rows, columns, neurons,
                                        kernel_size, kernel_size, weight_shift,
                                        neurons_prev_Layer);
}

static SbsLayer * SbsFullyConnectedLayer_new(uint16_t neurons,
                                                  uint16_t kernel_size,
                                                  WeightShift weight_shift,
                                                  uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (FULLY_CONNECTED_LAYER, 1, 1, neurons,
                                        kernel_size, 1, weight_shift,
                                        neurons_prev_Layer);
}

static SbsLayer * SbsOutputLayer_new(uint16_t neurons,
                                     WeightShift weight_shift,
                                     uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (OUTPUT_LAYER, 1, 1, neurons, 1, 1,
                                        weight_shift, neurons_prev_Layer);
}
/*****************************************************************************/

static SbsWeightMatrix SbsWeightMatrix_new(uint16_t rows, uint16_t columns, char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new(NULL, sizeof(Weight), 2, rows, columns);

    ASSERT(weight_watrix != NULL);
    ASSERT(weight_watrix->dimensionality == 2);
    ASSERT(weight_watrix->data != NULL);
    ASSERT(weight_watrix->dimension_size[0] == rows);
    ASSERT(weight_watrix->dimension_size[1] == columns);

    if ((weight_watrix != NULL)
        && (weight_watrix->dimensionality == 2)
        && (weight_watrix->data != NULL)
        && (weight_watrix->dimension_size[0] == rows)
        && (weight_watrix->dimension_size[1] == columns))
    {
#ifdef USE_XILINX
      FIL fil; /* File object */
      FRESULT rc;
      rc = f_open (&fil, file_name, FA_READ);
      ASSERT(rc == FR_OK);

      if (rc == FR_OK)
      {
        size_t read_size;
        size_t data_size = rows * columns * sizeof(Weight);
        rc = f_read (&fil, weight_watrix->data, data_size, &read_size);
        ASSERT((rc == FR_OK) && (read_size == data_size));
        f_close (&fil);
      }
      else Multivector_delete (&weight_watrix);
#else
      FILE * file = fopen(file_name, "rb");

      ASSERT(file != NULL);

      if (file != NULL)
      {
        size_t data_size = rows * columns * sizeof(Weight);
        size_t read_result = fread(weight_watrix->data, 1, data_size, file);
        ASSERT(data_size == read_result);
        fclose(file);
      }
      else
        Multivector_delete(&weight_watrix);
#endif
    }
  }

  return weight_watrix;
}

/*****************************************************************************/

SbsNetwork _SbsNetwork = {SbsBaseNetwork_new,
                          SbsBaseNetwork_delete,
                          SbsBaseNetwork_giveLayer,
                          SbsBaseNetwork_loadInput,
                          SbsBaseNetwork_updateCycle,
                          SbsBaseNetwork_getInferredOutput,
                          SbsBaseNetwork_getInputLabel,
                          SbsBaseNetwork_getOutputVector,
                          SbsBaseNetwork_getMemorySize};

SbsLayer _SbsLayer = {SbsBaseLayer_new,
                      SbsBaseLayer_delete,
                      SbsBaseLayer_setEpsilon,
                      SbsBaseLayer_setWeights};

SbsNew sbs_new = {SbsBaseNetwork_new,
                  SbsBaseLayer_new,
                  SbsWeightMatrix_new,
                  SbsInputLayer_new,
                  SbsConvolutionLayer_new,
                  SbsPoolingLayer_new,
                  SbsFullyConnectedLayer_new,
                  SbsOutputLayer_new};


/*****************************************************************************/
