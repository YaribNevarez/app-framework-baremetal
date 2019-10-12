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

#ifdef USE_XILINX
#include "ff.h"
#include "xparameters.h"
#include "xaxidma.h"

#ifdef USE_ACCELERATOR
#include "xsbs_update.h"
#include "xscugic.h"
#endif

#endif

#define ASSERT(expr)  assert(expr)

/*****************************************************************************/
#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */


typedef float     Weight;
typedef uint16_t  SpikeID;

typedef struct
{
  void *   data;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
} Multivector;

typedef struct
{
  SbsLayer      vtbl;

  Multivector * state_matrix;
  Multivector * weight_matrix;
  Multivector * spike_matrix;
  NeuronState * update_buffer;
  uint16_t      kernel_size;
  uint16_t      kernel_stride;
  uint16_t      neurons_previous_Layer;
  WeightShift   weight_shift;
  float         epsilon;
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
#define		MEMORY_SIZE         (4763116)

#define   MAX_LAYER_SIZE      (28*28)
#define   MAX_KERNEL_SIZE     (5*5)

#define   MAX_IP_VECTOR_SIZE  (1024)

#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
  #define       MEMORY_MGR_DDR_BASE_ADDRESS            (XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x01000000)
  #define       MEMORY_MGR_DDR_DMA_TX_BD_BASE_ADDRESS  (XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x02000000)
  #define       MEMORY_MGR_DDR_DMA_RX_BD_BASE_ADDRESS  (XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x03000000)

  #if MEMORY_MGR_DDR_DMA_TX_BD_BASE_ADDRESS < (MEMORY_MGR_DDR_BASE_ADDRESS + MEMORY_SIZE)
    #error "Overlapping memory-space and DMA-space"
  #endif
#endif

static size_t  Memory_blockIndex = 0;

static void * Memory_requestBlock(size_t size)
{
#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
  static uint8_t * Memory_block = (uint8_t *)MEMORY_MGR_DDR_BASE_ADDRESS;
#else
  static uint8_t Memory_block[MEMORY_SIZE];
#endif

  void * ptr = NULL;

  if (Memory_blockIndex + size <= MEMORY_SIZE)
  {
    ptr = (void *) &Memory_block[Memory_blockIndex];
    Memory_blockIndex += size;
  }

  return ptr;
}

static size_t Memory_getBlockSize(void)
{
  return Memory_blockIndex;
}

/*****************************************************************************/
/************************ Accelerator ****************************************/
#if defined(USE_XILINX) && defined(USE_ACCELERATOR)

#pragma pack(push)
#pragma pack(1)
typedef struct
{
  XSbs_update       updateHardware;

  XAxiDma           dmaHardware;
  XAxiDma_BdRing *  dmaRxBdRingPtr;
  XAxiDma_BdRing *  dmaTxBdRingPtr;
  XAxiDma_Bd *      dmaFirstTxBdPtr;
  XAxiDma_Bd *      dmaFirstRxBdPtr;
  XAxiDma_Bd *      dmaCurrentTxBdPtr;
  XAxiDma_Bd *      dmaCurrentRxBdPtr;
  uint16_t          txStateCounter;
  uint16_t          txWeightCounter;

  uint16_t          vectorSize;
  uint32_t          kernelSize;
  uint32_t          layerSize;

  uint8_t           errorFlags;
} SbSUpdateAccelerator;
#pragma pack(pop)

static SbSUpdateAccelerator Accelerator;
static XScuGic              ScuGic;

#define ACCELERATOR_DMA_RESET_TIMEOUT 10000

static void Accelerator_txInterruptHandler(void * data)
{
  u32 IrqStatus = XAxiDma_BdRingGetIrq(Accelerator.dmaTxBdRingPtr);

  XAxiDma_BdRingAckIrq(Accelerator.dmaTxBdRingPtr, IrqStatus);

  if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
  {
    return;
  }

  if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
  {
    int TimeOut;
    XAxiDma_BdRingDumpRegs (Accelerator.dmaTxBdRingPtr);

    Accelerator.errorFlags |= 0x01;

    XAxiDma_Reset (&Accelerator.dmaHardware);

    TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT;

    while (TimeOut)
    {
      if (XAxiDma_ResetIsDone (&Accelerator.dmaHardware))
      {
        break;
      }

      TimeOut -= 1;
    }

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

    BdCount = XAxiDma_BdRingFromHw(Accelerator.dmaTxBdRingPtr, XAXIDMA_ALL_BDS, &BdPtr);

    BdCurPtr = BdPtr;
    for (Index = 0; Index < BdCount; Index++)
    {
      BdSts = XAxiDma_BdGetSts(BdCurPtr);
      if ((BdSts & XAXIDMA_BD_STS_ALL_ERR_MASK) || (!(BdSts & XAXIDMA_BD_STS_COMPLETE_MASK)))
      {
        Accelerator.errorFlags |= 0x02;
        break;
      }

      BdCurPtr = (XAxiDma_Bd *) XAxiDma_BdRingNext(Accelerator.dmaTxBdRingPtr,
                                                   BdCurPtr);
    }

    status = XAxiDma_BdRingFree(Accelerator.dmaTxBdRingPtr, BdCount, BdPtr);
    ASSERT(status == XST_SUCCESS);
  }
}

static void Accelerator_rxInterruptHandler(void * data)
{
  u32 IrqStatus = XAxiDma_BdRingGetIrq(Accelerator.dmaRxBdRingPtr);

  XAxiDma_BdRingAckIrq(Accelerator.dmaRxBdRingPtr, IrqStatus);

  if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
  {
    return;
  }

  if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
  {
    int TimeOut;
    XAxiDma_BdRingDumpRegs (Accelerator.dmaRxBdRingPtr);

    Accelerator.errorFlags |= 0x01;

    XAxiDma_Reset (&Accelerator.dmaHardware);

    TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT;

    while (TimeOut)
    {
      if (XAxiDma_ResetIsDone (&Accelerator.dmaHardware))
      {
        break;
      }

      TimeOut -= 1;
    }

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

    BdCount = XAxiDma_BdRingFromHw(Accelerator.dmaRxBdRingPtr, XAXIDMA_ALL_BDS, &BdPtr);

    BdCurPtr = BdPtr;
    for (Index = 0; Index < BdCount; Index++)
    {
      BdSts = XAxiDma_BdGetSts(BdCurPtr);
      if ((BdSts & XAXIDMA_BD_STS_ALL_ERR_MASK) || (!(BdSts & XAXIDMA_BD_STS_COMPLETE_MASK)))
      {
        Accelerator.errorFlags |= 0x02;
        break;
      }

      BdCurPtr = (XAxiDma_Bd *) XAxiDma_BdRingNext(Accelerator.dmaRxBdRingPtr,
                                                   BdCurPtr);
    }

    status = XAxiDma_BdRingFree(Accelerator.dmaRxBdRingPtr, BdCount, BdPtr);
    ASSERT(status == XST_SUCCESS);
  }
}

static int Accelerator_initialize(void)
{
  XScuGic_Config *  IntcConfig;
  XAxiDma_Config *  dmaConfig;
  XAxiDma_Bd        dmaBdTemplate;
  u32               freeBdCount;
  int               status;


  if (MEMORY_MGR_DDR_DMA_RX_BD_BASE_ADDRESS
      < (MEMORY_MGR_DDR_DMA_TX_BD_BASE_ADDRESS
          + XAxiDma_BdRingMemCalc(XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                  MAX_LAYER_SIZE * MAX_KERNEL_SIZE)))
  {
    xil_printf ("Memory overlapping on TX_BD-space and RX_BD-space\n");

    return XST_FAILURE;
  }

  memset(&Accelerator, 0x00, sizeof(SbSUpdateAccelerator));

  /******************************* DMA initialization ************************/
#ifdef __aarch64__
  Xil_SetTlbAttributes(MEMORY_MGR_DDR_DMA_TX_BD_BASE_ADDRESS, MARK_UNCACHEABLE);
  Xil_SetTlbAttributes(MEMORY_MGR_DDR_DMA_RX_BD_BASE_ADDRESS, MARK_UNCACHEABLE);
#endif

  dmaConfig = XAxiDma_LookupConfig (XPAR_AXIDMA_0_DEVICE_ID);
  if (dmaConfig == NULL)
  {
    xil_printf ("No configuration found for %d\r\n", XPAR_AXIDMA_0_DEVICE_ID);

    return XST_FAILURE;
  }

  status = XAxiDma_CfgInitialize (&Accelerator.dmaHardware, dmaConfig);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Initialization failed %d\r\n", status);
    return XST_FAILURE;
  }

  if (!XAxiDma_HasSg(&Accelerator.dmaHardware))
  {
    xil_printf ("Device configured as Simple mode \r\n");

    return XST_FAILURE;
  }

  /**************************** DMA SG RX BD initialization ******************/
  Accelerator.dmaRxBdRingPtr = XAxiDma_GetRxRing(&Accelerator.dmaHardware);

  XAxiDma_BdRingIntEnable(Accelerator.dmaRxBdRingPtr, XAXIDMA_IRQ_ALL_MASK);

  XAxiDma_BdRingSetCoalesce (Accelerator.dmaRxBdRingPtr, 1, 0);

  status = XAxiDma_BdRingCreate (Accelerator.dmaRxBdRingPtr,
                                 MEMORY_MGR_DDR_DMA_RX_BD_BASE_ADDRESS,
                                 MEMORY_MGR_DDR_DMA_RX_BD_BASE_ADDRESS,
                                 XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                 MAX_LAYER_SIZE);

  if (status != XST_SUCCESS)
  {
    xil_printf ("RX create BD ring failed %d\r\n", status);

    return XST_FAILURE;
  }

  XAxiDma_BdClear(&dmaBdTemplate);

  status = XAxiDma_BdRingClone (Accelerator.dmaRxBdRingPtr, &dmaBdTemplate);
  if (status != XST_SUCCESS)
  {
    xil_printf ("RX clone BD failed %d\r\n", status);

    return XST_FAILURE;
  }

  freeBdCount = XAxiDma_BdRingGetFreeCnt(Accelerator.dmaRxBdRingPtr);

  if (freeBdCount != MAX_LAYER_SIZE)
  {
    xil_printf ("RX BD creation inconsistency\r\n");

    return XST_FAILURE;
  }

  status = XAxiDma_BdRingStart (Accelerator.dmaRxBdRingPtr);
  if (status != XST_SUCCESS)
  {
    xil_printf ("RX start hardware failed %d\r\n", status);

    return XST_FAILURE;
  }

  /**************************** DMA SG TX BD initialization ******************/
  Accelerator.dmaTxBdRingPtr = XAxiDma_GetTxRing(&Accelerator.dmaHardware);

  XAxiDma_BdRingIntEnable(Accelerator.dmaTxBdRingPtr, XAXIDMA_IRQ_ALL_MASK);

  XAxiDma_BdRingSetCoalesce(Accelerator.dmaTxBdRingPtr, 1, 0);


  status = XAxiDma_BdRingCreate (Accelerator.dmaTxBdRingPtr,
                                 MEMORY_MGR_DDR_DMA_TX_BD_BASE_ADDRESS,
                                 MEMORY_MGR_DDR_DMA_TX_BD_BASE_ADDRESS,
                                 XAXIDMA_BD_MINIMUM_ALIGNMENT,
                                 MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1));
  if (status != XST_SUCCESS)
  {
    xil_printf ("Failed to create BD ring in TX setup\r\n");

    return XST_FAILURE;
  }

  status = XAxiDma_BdRingClone (Accelerator.dmaTxBdRingPtr, &dmaBdTemplate);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Failed to BD ring clone in TX setup %d\r\n", status);

    return XST_FAILURE;
  }

  freeBdCount = XAxiDma_BdRingGetFreeCnt(Accelerator.dmaTxBdRingPtr);

  if (freeBdCount != MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1))
  {
    xil_printf ("RX BD creation inconsistency\r\n");

    return XST_FAILURE;
  }

  status = XAxiDma_BdRingStart (Accelerator.dmaTxBdRingPtr);
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
                                  XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID,
                                  0xA0, 0x3);

  XScuGic_SetPriorityTriggerType (&ScuGic,
                                  XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID,
                                  0xA0, 0x3);

  status = XScuGic_Connect (&ScuGic, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID,
                            (Xil_InterruptHandler) Accelerator_txInterruptHandler,
                            &Accelerator);
  if (status != XST_SUCCESS)
  {
    return status;
  }

  status = XScuGic_Connect (&ScuGic, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID,
                            (Xil_InterruptHandler) Accelerator_rxInterruptHandler,
                            &Accelerator);
  if (status != XST_SUCCESS)
  {
    return status;
  }

  XScuGic_Enable (&ScuGic, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID);
  XScuGic_Enable (&ScuGic, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID);

  /**************************** initialize ARM Core exception handlers *******/
  Xil_ExceptionInit ();
  Xil_ExceptionRegisterHandler (XIL_EXCEPTION_ID_INT,
                                (Xil_ExceptionHandler) XScuGic_InterruptHandler,
                                (void *) &ScuGic);

  Xil_ExceptionEnable();

  /***************************************************************************/
  /**************************** Accelerator initialization *******************/
  status = XSbs_update_Initialize (&Accelerator.updateHardware, XPAR_SBS_UPDATE_0_DEVICE_ID);
  if (status != XST_SUCCESS)
  {
    xil_printf ("Sbs update hardware initialization error: %d\r\n", status);

    return XST_FAILURE;
  }

  XSbs_update_InterruptGlobalDisable(&Accelerator.updateHardware);

  return XST_SUCCESS;
}

static void Accelerator_shutdown(void)
{
  XScuGic_Disconnect (&ScuGic, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID);
  XScuGic_Disconnect (&ScuGic, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID);
}

static void Accelerator_setup(NeuronState * layerData,
                              uint32_t      layerSize,
                              uint32_t      kernelSize,
                              uint16_t      vectorSize,
                              float         epsilon)
{
  int status;
  ASSERT (0 < layerSize);
  ASSERT (0 < kernelSize);
  ASSERT (0 < vectorSize);
  ASSERT (0.0 < epsilon);

  XSbs_update_Set_layerSize (&Accelerator.updateHardware, layerSize);
  Accelerator.layerSize = layerSize;

  XSbs_update_Set_kernelSize (&Accelerator.updateHardware, kernelSize);
  Accelerator.kernelSize = kernelSize;

  XSbs_update_Set_vectorSize (&Accelerator.updateHardware, vectorSize);
  Accelerator.vectorSize = vectorSize;

  XSbs_update_Set_epsilon (&Accelerator.updateHardware, *(uint32_t*) &epsilon);

  while (Accelerator.dmaTxBdRingPtr->PostCnt);
  while (Accelerator.dmaRxBdRingPtr->PostCnt);

  ASSERT(0 < XAxiDma_BdRingGetFreeCnt(Accelerator.dmaTxBdRingPtr));

  status = XAxiDma_BdRingAlloc (Accelerator.dmaTxBdRingPtr,
                                Accelerator.layerSize * (Accelerator.kernelSize + 1),
                                &Accelerator.dmaFirstTxBdPtr);
  ASSERT (status == XST_SUCCESS);

  XAxiDma_BdSetCtrl (Accelerator.dmaFirstTxBdPtr, XAXIDMA_BD_CTRL_TXSOF_MASK);

  Accelerator.dmaCurrentTxBdPtr = Accelerator.dmaFirstTxBdPtr;

  /************************** Rx Setup **************************/
  ASSERT(0 < XAxiDma_BdRingGetFreeCnt(Accelerator.dmaRxBdRingPtr));

  status = XAxiDma_BdRingAlloc (Accelerator.dmaRxBdRingPtr,
                                1,
                                &Accelerator.dmaFirstRxBdPtr);
  ASSERT(status == XST_SUCCESS);

  ASSERT (Accelerator.dmaFirstRxBdPtr != NULL);
  status = XAxiDma_BdSetBufAddr (Accelerator.dmaFirstRxBdPtr, (UINTPTR) layerData);
  ASSERT(status == XST_SUCCESS);

  ASSERT(0 < Accelerator.vectorSize);
  status = XAxiDma_BdSetLength (Accelerator.dmaFirstRxBdPtr,
                                layerSize * vectorSize * sizeof(NeuronState),
                                Accelerator.dmaRxBdRingPtr->MaxTransferLen);
  ASSERT(status == XST_SUCCESS);

  Accelerator.txStateCounter = 0;
  Accelerator.txWeightCounter = 0;
}

static void Accelerator_giveStateVector(NeuronState * state_vector)
{
  int status;

  ASSERT (state_vector != NULL);
  ASSERT (0 < Accelerator.vectorSize);

  Xil_DCacheFlushRange ((UINTPTR) state_vector, Accelerator.vectorSize * sizeof(NeuronState));
#ifdef __aarch64__
  /* TODO: Check cache flushing alignment */
  Xil_DCacheFlushRange ((UINTPTR) state_vector, Accelerator.vectorSize * sizeof(NeuronState));
#endif

  /************************** Tx Setup **************************/
  ASSERT (Accelerator.dmaCurrentTxBdPtr != NULL);
  status = XAxiDma_BdSetBufAddr (Accelerator.dmaCurrentTxBdPtr, (UINTPTR) state_vector);
  ASSERT (status == XST_SUCCESS);

  status = XAxiDma_BdSetLength (Accelerator.dmaCurrentTxBdPtr,
                                Accelerator.vectorSize * sizeof(NeuronState),
                                Accelerator.dmaTxBdRingPtr->MaxTransferLen);
  ASSERT (status == XST_SUCCESS);

  Accelerator.dmaCurrentTxBdPtr =
      (XAxiDma_Bd *) XAxiDma_BdRingNext(Accelerator.dmaTxBdRingPtr,
                                        Accelerator.dmaCurrentTxBdPtr);

  ASSERT (Accelerator.dmaCurrentTxBdPtr != NULL);

  Accelerator.txStateCounter ++;
  ASSERT(Accelerator.txStateCounter <= Accelerator.layerSize);
}

static void Accelerator_giveWeightVector (Weight * weight_vector)
{
  int status;

  ASSERT (weight_vector != NULL);

  Xil_DCacheFlushRange ((UINTPTR) weight_vector, Accelerator.vectorSize * sizeof(Weight));
#ifdef __aarch64__
  /* TODO: Check cache flushing memory alignment */
  Xil_DCacheFlushRange ((UINTPTR) weight_vector, Accelerator.vectorSize * sizeof(Weight));
#endif

  ASSERT (Accelerator.dmaCurrentTxBdPtr != NULL);
  ASSERT (Accelerator.dmaCurrentTxBdPtr != Accelerator.dmaFirstTxBdPtr);

  /* Set up the BD using the information of the packet to transmit */
  status = XAxiDma_BdSetBufAddr (Accelerator.dmaCurrentTxBdPtr, (UINTPTR) weight_vector);
  ASSERT(status == XST_SUCCESS);

  status = XAxiDma_BdSetLength (Accelerator.dmaCurrentTxBdPtr,
                                Accelerator.vectorSize * sizeof(Weight),
                                Accelerator.dmaTxBdRingPtr->MaxTransferLen);
  ASSERT(status == XST_SUCCESS);

  Accelerator.txWeightCounter ++;

  ASSERT(Accelerator.txWeightCounter <= Accelerator.kernelSize * Accelerator.layerSize);

  if (Accelerator.txWeightCounter < Accelerator.kernelSize * Accelerator.layerSize)
  {
    Accelerator.dmaCurrentTxBdPtr =
        (XAxiDma_Bd *) XAxiDma_BdRingNext(Accelerator.dmaTxBdRingPtr,
                                          Accelerator.dmaCurrentTxBdPtr);

    ASSERT(Accelerator.dmaCurrentTxBdPtr != NULL);
  }
  else
  {
    ASSERT(Accelerator.txWeightCounter == Accelerator.kernelSize * Accelerator.layerSize);
    XAxiDma_BdSetCtrl (Accelerator.dmaCurrentTxBdPtr, XAXIDMA_BD_CTRL_TXEOF_MASK);
  }
}

static void Accelerator_start(void)
{
  uint32_t allocatedDmaBd;
  int status;

  while (!XSbs_update_IsReady (&Accelerator.updateHardware));

  XSbs_update_Start (&Accelerator.updateHardware);

  allocatedDmaBd = Accelerator.layerSize * (Accelerator.kernelSize + 1);

  ASSERT (allocatedDmaBd == (Accelerator.txStateCounter + Accelerator.txWeightCounter));
  ASSERT (Accelerator.layerSize == Accelerator.txStateCounter);

  status = XAxiDma_BdRingToHw (Accelerator.dmaTxBdRingPtr,
                               allocatedDmaBd,
                               Accelerator.dmaFirstTxBdPtr);
  ASSERT(status == XST_SUCCESS);

  status = XAxiDma_BdRingToHw (Accelerator.dmaRxBdRingPtr,
                               1,
                               Accelerator.dmaFirstRxBdPtr);
  ASSERT(status == XST_SUCCESS);

//  Xil_DCacheInvalidateRange ((UINTPTR) Accelerator.stateVector,
//                             Accelerator.vectorSize * sizeof(NeuronState));

  Accelerator.dmaFirstTxBdPtr = NULL;
  Accelerator.dmaFirstRxBdPtr = NULL;
  Accelerator.dmaCurrentTxBdPtr = NULL;
  Accelerator.dmaCurrentRxBdPtr = NULL;
}
#endif

/*****************************************************************************/
/*****************************************************************************/

static Multivector * Multivector_new(uint8_t data_type_size, uint8_t dimensionality, ...)
{
  Multivector * multivector = NULL;

  ASSERT(0 <= dimensionality);

  if (0 <= dimensionality)
  {
    size_t memory_size = sizeof(Multivector) + (dimensionality - 1) * sizeof(uint16_t);
    multivector = malloc(memory_size);

    ASSERT(multivector != NULL);

    if (multivector != NULL)
    {
      int arg;
      size_t data_size;
      va_list argument_list;

      memset(multivector, 0x00, memory_size);

      va_start(argument_list, dimensionality);

      for (data_size = 1, arg = 0; arg < dimensionality; arg ++)
        data_size *= (multivector->dimension_size[arg] = (uint16_t) va_arg(argument_list, int));

      va_end(argument_list);

      multivector->data = Memory_requestBlock(data_size * data_type_size);

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size * data_type_size);

      multivector->dimensionality = dimensionality;
    }
  }

  return multivector;
}

static void Multivector_delete(Multivector ** multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(*multivector != NULL);

  if ((multivector != NULL) && (*multivector != NULL))
  {
    free(*multivector);
    *multivector = NULL;
  }
}
/*****************************************************************************/
/*****************************************************************************/

static SbsLayer * SbsBaseLayer_new(uint16_t rows,
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
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    /* Instantiate state_matrix */
    state_matrix = Multivector_new(sizeof(NeuronState), 3, rows, columns, neurons);

    ASSERT(state_matrix != NULL);
    ASSERT(state_matrix->dimensionality == 3);
    ASSERT(state_matrix->data != NULL);
    ASSERT(state_matrix->dimension_size[0] == rows);
    ASSERT(state_matrix->dimension_size[1] == columns);
    ASSERT(state_matrix->dimension_size[2] == neurons);

    layer->state_matrix = state_matrix;

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new(sizeof(SpikeID), 2, rows, columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    layer->spike_matrix = spike_matrix;

    /* Allocate update buffer */

    layer->update_buffer = malloc(neurons * sizeof(NeuronState));

    ASSERT(layer->update_buffer != NULL);

    if (layer->update_buffer != NULL)
    	memset(layer->update_buffer, 0x00, neurons * sizeof(NeuronState));

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
  if ((layer_ptr!= NULL) && (*layer_ptr!= NULL))
  {
    SbsBaseLayer ** layer = (SbsBaseLayer **)layer_ptr;
    Multivector_delete(&((*layer)->state_matrix));
    Multivector_delete(&((*layer)->spike_matrix));
    if ((*layer)->weight_matrix != NULL) Multivector_delete(&((*layer)->weight_matrix));
    free((*layer)->update_buffer);
    free(*layer);
    *layer = NULL;
  }
}


static void SbsBaseLayer_initializeIP(NeuronState * state_vector, uint16_t size)
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

#if !defined(USE_ACCELERATOR)
static void SbsBaseLayer_updateIP(SbsBaseLayer * layer, NeuronState * state_vector, Weight * weight_vector, uint16_t size, float epsilon)
{
  ASSERT(state_vector != NULL);
  ASSERT(weight_vector != NULL);
  ASSERT(layer->update_buffer != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (weight_vector != NULL)
      && (layer->update_buffer != NULL) && (0 < size))
  {
    NeuronState * temp_data     = layer->update_buffer;

    NeuronState sum             = 0.0f;
    NeuronState reverse_epsilon = 0.0f;
    NeuronState epsion_over_sum = 0.0f;
    uint16_t    neuron;

#if defined (__x86_64__) || defined(__amd64__)
    {
      for (neuron = 0; neuron < size; neuron++)
      {
        temp_data[neuron] = state_vector[neuron] * weight_vector[neuron];
        sum += temp_data[neuron];
      }

      if (1e-20 < sum) // TODO: DEFINE constant
      {
        epsion_over_sum = epsilon / sum;
        reverse_epsilon = 1.0f / (1.0f + epsilon);

        for (neuron = 0; neuron < size; neuron++)
          state_vector[neuron] = reverse_epsilon
              * (state_vector[neuron] + temp_data[neuron] * epsion_over_sum);
      }
    }
#elif defined(__arm__)
    {
      /* Support for unaligned accesses in ARM architecture */
      NeuronState h;
      NeuronState p;
      NeuronState h_p;
      NeuronState h_new;

      for (neuron = 0; neuron < size; neuron++)
      {
        h = state_vector[neuron];
        p = weight_vector[neuron];
        h_p = h * p;

        temp_data[neuron] = h_p;
        sum += h_p;
      }

      if (1e-20 < sum) // TODO: DEFINE constant
      {
        epsion_over_sum = epsilon / sum;
        reverse_epsilon = 1.0f / (1.0f + epsilon);

        for (neuron = 0; neuron < size; neuron++)
        {
          h_p = temp_data[neuron];
          h = state_vector[neuron];

          h_new = reverse_epsilon * (h + h_p * epsion_over_sum);
          state_vector[neuron] = h_new;
        }
      }
    }
#else
#error "Unsupported processor architecture"
#endif
  }
}
#endif

static SpikeID SbsBaseLayer_generateSpikeIP (NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    NeuronState random_s = ((NeuronState) genrand ()) / ((NeuronState) 0xFFFFFFFF);
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

static void SbsBaseLayer_initialize(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->state_matrix != NULL);
  ASSERT(layer->state_matrix->data != NULL);

  if ((layer != NULL) && (layer->state_matrix != NULL) && (layer->state_matrix->data != NULL))
  {
    Multivector * state_matrix      = layer->state_matrix;
    uint16_t      rows              = state_matrix->dimension_size[0];
    uint16_t      columns           = state_matrix->dimension_size[1];
    uint16_t      neurons           = state_matrix->dimension_size[2];
    NeuronState * state_matrix_data = state_matrix->data;

    uint16_t row;
    uint16_t column;
    size_t   current_row_index;

    for (row = 0; row < rows; row++)
    {
      current_row_index = row * columns * neurons;
      for (column = 0; column < columns; column++)
      {
        SbsBaseLayer_initializeIP(&state_matrix_data[current_row_index + column * neurons], neurons);
      }
    }
  }
}

static void SbsBaseLayer_giveWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->weight_matrix = (Multivector *) weight_matrix;
}

static void SbsBaseLayer_setEpsilon(SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->epsilon = epsilon;
}

static Multivector * SbsBaseLayer_generateSpikes(SbsBaseLayer * layer)
{
  Multivector * spike_matrix = NULL;
  ASSERT(layer != NULL);
  ASSERT(layer->state_matrix != NULL);
  ASSERT(layer->spike_matrix != NULL);
  ASSERT(layer->state_matrix->data != NULL);
  ASSERT(layer->spike_matrix->data != NULL);

  if (   (layer != NULL)
      && (layer->state_matrix != NULL)
      && (layer->spike_matrix != NULL)
      && (layer->state_matrix->data != NULL)
      && (layer->spike_matrix->data != NULL))
  {
      Multivector * state_matrix      = layer->state_matrix;
      uint16_t      rows              = state_matrix->dimension_size[0];
      uint16_t      columns           = state_matrix->dimension_size[1];
      uint16_t      neurons           = state_matrix->dimension_size[2];
      NeuronState * state_matrix_data = state_matrix->data;
      SpikeID *     spike_matrix_data = layer->spike_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t   current_row_index;
      size_t   current_row_column_index;

      for (row = 0; row < rows; row++)
      {
        current_row_index = columns * row;
        for (column = 0; column < columns; column++)
        {
            current_row_column_index = current_row_index + column;
            spike_matrix_data[current_row_column_index] = SbsBaseLayer_generateSpikeIP(&state_matrix_data[current_row_column_index * neurons], neurons);
        }
      }

      spike_matrix = layer->spike_matrix;
  }

  return spike_matrix;
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

    ASSERT(weight_columns == neurons);

    if (weight_columns != neurons)
      return;

    if (layer->weight_shift == ROW_SHIFT)
    {
      row_shift = 1;
      column_shift = kernel_size;
    }

#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
    Accelerator_setup (state_data,
                       state_matrix->dimension_size[0] * state_matrix->dimension_size[1],
                       kernel_size * kernel_size,
                       neurons,
                       epsilon);
#endif

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

#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
        Accelerator_giveStateVector (state_vector);
#endif
        for (kernel_row = 0; kernel_row < kernel_size; kernel_row ++)
        {
            spike_row_index = (kernel_row_pos + kernel_row) * spike_columns;
          for (kernel_column = 0; kernel_column < kernel_size; kernel_column ++)
          {
            spikeID = spike_data[spike_row_index + kernel_column_pos + kernel_column];

            section_shift = (kernel_row * row_shift + kernel_column * column_shift) * neurons_previous_Layer;

            weight_vector = &weight_data[(spikeID + section_shift) * weight_columns];

#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
            Accelerator_giveWeightVector (weight_vector);
#else
            SbsBaseLayer_updateIP (layer, state_vector, weight_vector, neurons, epsilon);
#endif
          }
        }
      }
    }
#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
        Accelerator_start();
#endif
    /* Update ends*/
  }
}

/*****************************************************************************/

static SbsNetwork * SbsBaseNetwork_new(void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc(sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
      memset(network, 0x0, sizeof(SbsBaseNetwork));
      network->vtbl = _SbsNetwork;
      network->input_label = (uint8_t)-1;
      network->inferred_output = (uint8_t)-1;

#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
      ASSERT(Accelerator_initialize() == XST_SUCCESS); /* TODO: Create interface for Accelerator */
#endif
      sgenrand(666); /*TODO: Create MT19937 object wrapper */
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

#if defined(USE_XILINX) && defined(USE_ACCELERATOR)
  Accelerator_shutdown (); /* TODO: Create interface for Accelerator */
#endif
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
        if (i < network->size - 1)
          SbsBaseLayer_generateSpikes(network->layer_array[i]);

        if (0 < i)
          SbsBaseLayer_update(network->layer_array[i],
              network->layer_array[i - 1]->spike_matrix);
      }

      if (cycles % 100 == 0)
        printf(" - Spike cycle: %d\n", cycles);
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
  return Memory_getBlockSize();
}
/*****************************************************************************/

static SbsLayer * SbsInputLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new(rows, columns, neurons, 0, 0, ROW_SHIFT, 0);
}

static SbsLayer * SbsConvolutionLayer_new(uint16_t rows,
                                            uint16_t columns,
                                            uint16_t neurons,
                                            uint16_t kernel_size,
                                            WeightShift weight_shift,
                                            uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (rows, columns, neurons, kernel_size, 1, weight_shift,
                           neurons_prev_Layer);
}

static SbsLayer * SbsPoolingLayer_new(uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift,
                                    uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (rows, columns, neurons, kernel_size,
                                        kernel_size, weight_shift, neurons_prev_Layer);
}

static SbsLayer * SbsFullyConnectedLayer_new(uint16_t neurons,
                                                  uint16_t kernel_size,
                                                  WeightShift weight_shift,
                                                  uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (1, 1, neurons, kernel_size, 1,
                                        weight_shift, neurons_prev_Layer);
}

static SbsLayer * SbsOutputLayer_new(uint16_t neurons,
                                  WeightShift weight_shift,
                                  uint16_t neurons_prev_Layer)
{
  return (SbsLayer *) SbsBaseLayer_new (1, 1, neurons, 1, 1, weight_shift,
                                        neurons_prev_Layer);
}
/*****************************************************************************/

static SbsWeightMatrix SbsWeightMatrix_new(uint16_t rows, uint16_t columns, char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new(sizeof(Weight), 2, rows, columns);

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
                      SbsBaseLayer_giveWeights};

SbsNew sbs_new = {SbsBaseNetwork_new,
                  SbsBaseLayer_new,
                  SbsWeightMatrix_new,
                  SbsInputLayer_new,
                  SbsConvolutionLayer_new,
                  SbsPoolingLayer_new,
                  SbsFullyConnectedLayer_new,
                  SbsOutputLayer_new};


/*****************************************************************************/
