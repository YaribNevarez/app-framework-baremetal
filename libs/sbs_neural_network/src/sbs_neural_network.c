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
#include "xtime_l.h"


#include "xsbs_update.h"
#include "xscugic.h"

//#define DEBUG

#ifdef DEBUG

void sbs_assert(const char * file, int line, const char * function, const char * expression)
{
  printf ("Fail: %s, in \"%s\" [%s, %d]\n", expression, function, file, line);

  for (;;);
}

#define ASSERT(expr) if (!(expr)) sbs_assert(__FILE__, __LINE__, __func__, #expr);
#else
#define ASSERT(expr)
#endif

/*****************************************************************************/
#define   MEMORY_SIZE         (4771384)
#define   MAX_LAYER_SIZE      (28*28)
#define   MAX_KERNEL_SIZE     (5*5)
#define   MAX_IP_VECTOR_SIZE  (1024)  // Inference population size
#define   MAX_NETWORK_SIZE    (7)     // MAX number of layers in a network

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
  uint32_t    updateIntVecID;
  uint32_t    dmaTxIntVecID;
  uint32_t    dmaRxIntVecID;
  MemoryBlock ddrMem;
} SbSHardwareConfig;

typedef struct
{
  uint32_t    layerSize;
  uint32_t    kernelSize;
  uint32_t    vectorSize;
  float       epsilon;

  sigset_t    vectorBufferSize;

  size_t      txBufferSize;
  uint32_t *  rxBuffer;
  size_t      rxBufferSize;
} SbsAcceleratorProfie;

typedef struct
{
  SbSHardwareConfig *     hardwareConfig;
  XSbs_update             updateHardware;
  XAxiDma                 dmaHardware;
  SbsAcceleratorProfie *  profile;

#ifdef DEBUG
  uint16_t            txStateCounter;
  uint16_t            txWeightCounter;
#endif

  uint8_t           txDone;
  uint8_t           rxDone;
  uint8_t           acceleratorReady;

  uint32_t          txBufferIndex;
  uint32_t *        txBuffer;
  size_t            txBufferSize;

  uint32_t *        rxBuffer;
  size_t            rxBufferSize;

  uint8_t           errorFlags;
} SbSUpdateAccelerator;


typedef float     Weight;
typedef uint32_t  SpikeID;

typedef enum
{
  M32BIT_TYPE_BEGIN = 0,
  M32BIT_24_24_ID,
  M32BIT_24_24_50_ID,
  M32BIT_12_24_32_ID,
  M32BIT_1_1_50_32_ID,
  M32BIT_6_12_32_ID,
  M32BIT_12_12_ID,
  M32BIT_2_2_32_32_ID,
  M32BIT_8_8_64_ID,
  M32BIT_8_8_ID,
  M32BIT_5_5_32_64_ID,
  M32BIT_2_4_64_ID,
  M32BIT_4_4_ID,
  M32BIT_2_2_64_64_ID,
  M32BIT_1_1_1024_ID,
  M32BIT_1_1_ID,
  M32BIT_4_4_64_1024_ID,
  M32BIT_1_1_10_ID,
  M32BIT_1_1_1024_10_ID,
  M32BIT_TYPE_END = (unsigned)-1
} M32BitTypeID;

typedef uint32_t M32Bit_24_24[24][24];
typedef uint32_t M32Bit_24_24_50[24][24][50];
typedef uint32_t M32Bit_12_24_32[12][24][32];
typedef uint32_t M32Bit_1_1_50_32[1][1][50][32];
typedef uint32_t M32Bit_6_12_32[6][12][32];
typedef uint32_t M32Bit_12_12[12][12];
typedef uint32_t M32Bit_2_2_32_32[2][2][32][32];
typedef uint32_t M32Bit_8_8_64[8][8][64];
typedef uint32_t M32Bit_8_8[8][8];
typedef uint32_t M32Bit_5_5_32_64[5][5][32][64];
typedef uint32_t M32Bit_2_4_64[2][4][64];
typedef uint32_t M32Bit_4_4[4][4];
typedef uint32_t M32Bit_2_2_64_64[2][2][64][64];
typedef uint32_t M32Bit_1_1_1024[1][1][1024];
typedef uint32_t M32Bit_1_1[1][1];
typedef uint32_t M32Bit_4_4_64_1024[4][4][64][1024];
typedef uint32_t M32Bit_1_1_10[1][1][10];
typedef uint32_t M32Bit_1_1_1024_10[1][1][1024][10];

typedef struct
{
  M32BitTypeID type_id;
  uint8_t data_type_size;
  uint8_t dimensionality;
  uint16_t dimension_size[4];
} M32BitFormat;

M32BitFormat M32BitFormat_list[] =
{
    {
        .type_id = M32BIT_24_24_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {24, 24, 0, 0}
    },
    {
        .type_id = M32BIT_24_24_50_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {24, 24, 50, 0}
    },
    {
        .type_id = M32BIT_12_24_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {12, 24, 32, 0}
    },
    {
        .type_id = M32BIT_1_1_50_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 4,
        .dimension_size = {1, 1, 50, 32}
    },
    {
        .type_id = M32BIT_6_12_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {6, 12, 32, 0}
    },
    {
        .type_id = M32BIT_12_12_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {12, 12, 0, 0}
    },
    {
        .type_id = M32BIT_2_2_32_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 4,
        .dimension_size = {2, 2, 32, 32}
    },
    {
        .type_id = M32BIT_8_8_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {8, 8, 64, 0}
    },
    {
        .type_id = M32BIT_8_8_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {8, 8, 0, 0}
    },
    {
        .type_id = M32BIT_5_5_32_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 4,
        .dimension_size = {5, 5, 32, 64}
    },
    {
        .type_id = M32BIT_2_4_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {2, 4, 64, 0}
    },
    {
        .type_id = M32BIT_4_4_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {4, 4, 0, 0}
    },
    {
        .type_id = M32BIT_2_2_64_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 4,
        .dimension_size = {2, 2, 64, 64}
    },
    {
        .type_id = M32BIT_1_1_1024_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {1, 1, 1024, 0}
    },
    {
        .type_id = M32BIT_1_1_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {1, 1, 0, 0}
    },
    {
        .type_id = M32BIT_4_4_64_1024_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 4,
        .dimension_size = {4, 4, 64, 1024}
    },
    {
        .type_id = M32BIT_1_1_10_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {1, 1, 10, 0}
    },
    {
        .type_id = M32BIT_1_1_1024_10_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 4,
        .dimension_size = {1, 1, 1024, 10}
    }
};

const unsigned M32BitFormat_list_length = (sizeof(M32BitFormat_list) / sizeof (M32BitFormat));

M32BitTypeID M32BitFormat_getTypeID(uint8_t data_type_size, uint8_t dimensionality, uint16_t * dimension_size)
{
  int i;
  M32BitTypeID type_ID = M32BIT_TYPE_END;

  for (i = 0; i < M32BitFormat_list_length; i++)
    if (M32BitFormat_list[i].data_type_size == data_type_size
        && M32BitFormat_list[i].dimensionality == dimensionality
        && 0 == memcmp (M32BitFormat_list[i].dimension_size,
                        dimension_size,
                        dimensionality * sizeof(uint16_t)))
      type_ID = M32BitFormat_list[i].type_id;

  ASSERT (type_ID != M32BIT_TYPE_END);

  return type_ID;
}

typedef struct
{
  MemoryBlock * memory_def_parent;

  void *   data;
  M32BitTypeID type_id;
  uint8_t  data_type_size;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
} Multivector;

typedef struct
{
  SbSUpdateAccelerator *  accelerator;
  SbsAcceleratorProfie    profile;
  uint16_t      x_pos;
  uint16_t      y_pos;
  Multivector * state_matrix;
  Multivector * weight_matrix;
} SbsLayerPartition;

typedef struct
{
  SbsLayer              vtbl;
  SbsLayerType          layer_type;
  SbsLayerPartition **  partition_array;
  uint8_t               num_partitions;
  Multivector *         spike_matrix;
  uint16_t              rows;
  uint16_t              columns;
  uint16_t              neurons;
  uint16_t              kernel_size;
  uint16_t              kernel_stride;
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

typedef struct
{
  XTime   start_time;
  uint8_t num_samples;
  XTime   sample_array[1];
} Timer;

#pragma pack(pop)   /* restore original alignment from stack */

/*****************************************************************************/
/************************ Timer **********************************************/

Timer * Timer_new (uint8_t num_samples)
{
  Timer * timer = NULL;
  ASSERT(0 < num_samples);
  if (0 < num_samples)
  {
    size_t size = sizeof(Timer) + ((num_samples - 1) * sizeof(XTime));
    timer = malloc (size);
    ASSERT(timer != NULL);
    if (timer != NULL)
    {
      memset (timer, 0x00, size);
      timer->num_samples = num_samples;
    }
  }

  return timer;
}

void Timer_delete (Timer ** timer)
{
  ASSERT (timer != NULL);
  ASSERT (*timer != NULL);

  if ((timer != NULL) && (*timer != NULL))
  {
    free (*timer);
    *timer = NULL;
  }
}

void Timer_start (Timer * timer)
{
  ASSERT(timer != NULL);
  if (timer != NULL)
    XTime_GetTime (&timer->start_time);
}

double Timer_getCurrentTime (Timer * timer)
{
  double time = 0.0;
  ASSERT(timer != NULL);
  if (timer != NULL)
  {
    XTime temp;
    XTime_GetTime (&temp);
    time = ((double) (temp - timer->start_time)) / ((double) COUNTS_PER_SECOND);
  }
  return time;
}

void Timer_takeSample (Timer * timer, uint8_t index, double * sample)
{
  ASSERT(timer != NULL);
  ASSERT(index < timer->num_samples);
  if ((timer != NULL) && (index < timer->num_samples))
  {
    XTime_GetTime (&timer->sample_array[index]);
    if (sample != NULL)
      *sample = ((double) (timer->sample_array[index] - timer->start_time))
          / ((double) COUNTS_PER_SECOND);
  }
}

double Timer_getSample(Timer * timer, uint8_t index)
{
  double sample = 0.0;
  ASSERT(timer != NULL);
  ASSERT(index < timer->num_samples);
  if ((timer != NULL) && (index < timer->num_samples))
    sample = ((double) (timer->sample_array[index] - timer->start_time))
                      / ((double) COUNTS_PER_SECOND);
  return sample;
}

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

  ASSERT (ptr != NULL);

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
    .dmaDeviceID = XPAR_AXIDMA_0_DEVICE_ID,
    .updateIntVecID = XPAR_FABRIC_SBS_UPDATE_0_VEC_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID,
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
    .dmaDeviceID = XPAR_AXIDMA_1_DEVICE_ID,
    .updateIntVecID = XPAR_FABRIC_SBS_UPDATE_1_VEC_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_1_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_1_S2MM_INTROUT_VEC_ID,
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
    .dmaDeviceID = XPAR_AXIDMA_2_DEVICE_ID,
    .updateIntVecID = XPAR_FABRIC_SBS_UPDATE_2_VEC_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_2_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_2_S2MM_INTROUT_VEC_ID,
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
    .dmaDeviceID = XPAR_AXIDMA_3_DEVICE_ID,
    .updateIntVecID = XPAR_FABRIC_SBS_UPDATE_3_VEC_ID,
    .dmaTxIntVecID = XPAR_FABRIC_AXIDMA_3_MM2S_INTROUT_VEC_ID,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_3_S2MM_INTROUT_VEC_ID,
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
  XAxiDma *AxiDmaInst = &((SbSUpdateAccelerator *) data)->dmaHardware;
  u32 IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DMA_TO_DEVICE);

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
  XAxiDma *AxiDmaInst = &((SbSUpdateAccelerator *) data)->dmaHardware;
  u32 IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DEVICE_TO_DMA);

  XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DEVICE_TO_DMA);

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
    ((SbSUpdateAccelerator *) data)->rxDone = 1;
  }
}

static void Accelerator_updateInterruptHandler (void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  uint32_t status;

  status = XSbs_update_InterruptGetStatus(&accelerator->updateHardware);
  XSbs_update_InterruptClear(&accelerator->updateHardware, status);
  accelerator->acceleratorReady = status & 1;
}

static int Accelerator_initialize(SbSUpdateAccelerator * accelerator, SbSHardwareConfig * hardware_config)
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

  if (XAxiDma_HasSg(&accelerator->dmaHardware))
  {
    xil_printf ("Device configured as SG mode \r\n");

    return XST_FAILURE;
  }

  XAxiDma_IntrEnable(&accelerator->dmaHardware,
                     XAXIDMA_IRQ_ALL_MASK,
                     XAXIDMA_DMA_TO_DEVICE);

  XAxiDma_IntrEnable(&accelerator->dmaHardware,
                     XAXIDMA_IRQ_ALL_MASK,
                     XAXIDMA_DEVICE_TO_DMA);

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

  XScuGic_SetPriorityTriggerType (&ScuGic,
                                  hardware_config->updateIntVecID,
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

  status = XScuGic_Connect (&ScuGic, hardware_config->updateIntVecID,
                            (Xil_InterruptHandler) Accelerator_updateInterruptHandler,
                            accelerator);
  if (status != XST_SUCCESS)
  {
    return status;
  }

  XScuGic_Enable (&ScuGic, hardware_config->dmaTxIntVecID);
  XScuGic_Enable (&ScuGic, hardware_config->dmaRxIntVecID);
  XScuGic_Enable (&ScuGic, hardware_config->updateIntVecID);

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

  XSbs_update_InterruptGlobalEnable (&accelerator->updateHardware);
  XSbs_update_InterruptEnable (&accelerator->updateHardware, 1);
  accelerator->acceleratorReady = 1;
  accelerator->rxDone = 1;
  accelerator->txDone = 1;

  /***************************************************************************/
  accelerator->txBuffer = MemoryBlock_alloc(&hardware_config->ddrMem,
                                          (MAX_LAYER_SIZE/2) * (MAX_KERNEL_SIZE + 1) * MAX_IP_VECTOR_SIZE * sizeof(NeuronState));

  ASSERT (accelerator->txBuffer != NULL);

  if (accelerator->txBuffer == NULL)
  {
    xil_printf ("DMA TX buffer allocation error\r\n");

    return XST_FAILURE;
  }

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
    XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->updateIntVecID);
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

static void Accelerator_setup(SbSUpdateAccelerator * accelerator,
                              SbsAcceleratorProfie * profile)
{
  ASSERT (accelerator != NULL);
  ASSERT (profile != NULL);

  if (accelerator->profile != profile)
  {
    accelerator->profile = profile;

    XSbs_update_Set_layerSize (&accelerator->updateHardware,
                               profile->layerSize);

    XSbs_update_Set_kernelSize (&accelerator->updateHardware,
                                profile->kernelSize);

    XSbs_update_Set_vectorSize (&accelerator->updateHardware,
                                profile->vectorSize);

    XSbs_update_Set_epsilon (&accelerator->updateHardware,
                             *(uint32_t*) &profile->epsilon);

    /************************** Tx Setup **************************/
    ASSERT(accelerator->txBuffer != NULL);
    accelerator->txBufferSize = profile->txBufferSize;

    /************************** Rx Setup **************************/
    accelerator->rxBuffer = profile->rxBuffer;
    accelerator->rxBufferSize = profile->rxBufferSize;
  }

  accelerator->txBufferIndex = 0;

#ifdef DEBUG
  accelerator->txStateCounter = 0;
  accelerator->txWeightCounter = 0;
#endif
}

inline static void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         NeuronState * state_vector) __attribute__((always_inline));

inline static void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         NeuronState * state_vector)
{
  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->layerSize);
  ASSERT (0 < accelerator->profile->vectorBufferSize);
  ASSERT (state_vector != NULL);

  memcpy(&accelerator->txBuffer[accelerator->txBufferIndex],
         state_vector,
         accelerator->profile->vectorBufferSize);


  accelerator->txBufferIndex += accelerator->profile->vectorSize;

  ASSERT(accelerator->txStateCounter <= accelerator->profile->layerSize);

#ifdef DEBUG
  accelerator->txStateCounter ++;
#endif
}

inline static void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          Weight * weight_vector) __attribute__((always_inline));

inline static void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          Weight * weight_vector)
{
  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->vectorBufferSize);
  ASSERT (0 < accelerator->profile->kernelSize);
  ASSERT (0 < accelerator->profile->layerSize);
  ASSERT (weight_vector != NULL);

  ASSERT(accelerator->txWeightCounter <= accelerator->profile->kernelSize * accelerator->profile->layerSize);

  memcpy(&accelerator->txBuffer[accelerator->txBufferIndex],
         weight_vector,
         accelerator->profile->vectorBufferSize);

  accelerator->txBufferIndex += accelerator->profile->vectorSize;

#ifdef DEBUG
  accelerator->txWeightCounter ++;
#endif
}

int accelerator_wait [7][4] = {0};
int tx_wait [7][4] = {0};
int rx_wait [7][4] = {0};
int layer_wait = 0;

static void Accelerator_start(SbSUpdateAccelerator * accelerator)
{
  int status;

  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->vectorBufferSize);
  ASSERT (0 < accelerator->profile->kernelSize);
  ASSERT (0 < accelerator->profile->layerSize);

#ifdef DEBUG
  ASSERT (accelerator->profile->layerSize == accelerator->txStateCounter);
  ASSERT (accelerator->profile->kernelSize * accelerator->profile->layerSize == accelerator->txWeightCounter);
#endif

  Xil_DCacheFlushRange ((UINTPTR) accelerator->txBuffer, accelerator->txBufferSize);

  while (accelerator->txDone == 0) tx_wait[layer_wait][accelerator->hardwareConfig->updateDeviceID] ++;
  status = XAxiDma_SimpleTransfer (&accelerator->dmaHardware,
                                   (UINTPTR) accelerator->txBuffer,
                                   accelerator->txBufferSize,
                                   XAXIDMA_DMA_TO_DEVICE);
  ASSERT(status == XST_SUCCESS);
  accelerator->txDone = 0;

  while (accelerator->rxDone == 0) rx_wait[layer_wait][accelerator->hardwareConfig->updateDeviceID] ++;
  status = XAxiDma_SimpleTransfer (&accelerator->dmaHardware,
                                   (UINTPTR) accelerator->rxBuffer,
                                   accelerator->rxBufferSize,
                                   XAXIDMA_DEVICE_TO_DMA);
  ASSERT(status == XST_SUCCESS);
  accelerator->rxDone = 0;

  while (accelerator->acceleratorReady == 0) accelerator_wait[layer_wait][accelerator->hardwareConfig->updateDeviceID] ++;
  XSbs_update_Start (&accelerator->updateHardware);
  accelerator->acceleratorReady = 0;
}

/*****************************************************************************/

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
    for (i = 0; i < NUM_ACCELERATOR_INSTANCES; i++)
    {
      Accelerator_delete ((&Accelerator_array[i]));
    }

    free (Accelerator_array);
  }
}

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
      multivector->type_id = M32BitFormat_getTypeID(data_type_size,
                                                    dimensionality,
                                                    multivector->dimension_size);
    }
  }

  return multivector;
}

//Multivector * multivector_array[80] = { 0 };
//int multivector_array_count = 0;
//
//void MultivectorArray_add(Multivector * multivector)
//{
//  int i;
//  for (i = 0;
//      i < multivector_array_count && multivector_array[i] != multivector;
//      i++);
//
//  if (i == multivector_array_count)
//  {
//    for (int t = 0; t < multivector_array_count; t ++)
//      if (multivector_array[t]->dimensionality == multivector->dimensionality)
//      {
//        int d;
//        for (d = 0; d < multivector_array[t]->dimensionality && (multivector_array[t]->dimension_size[d] == multivector->dimension_size[d]); d ++);
//
//        if (d == multivector_array[t]->dimensionality && multivector_array[t]->data_type_size == multivector->data_type_size)
//          return;
//      }
//
//    multivector_array[i] = multivector;
//    multivector_array_count ++;
//  }
//}
//
//int str_len (char * str)
//{
//  int i = 0;
//  while (str[i] != 0)
//    i++;
//  return i;
//}
//
//void MultivectorArray_print()
//{
//  int i;
//  int d;
//  char text[900] = {0};
//  for (i = 0; i < multivector_array_count; i++)
//  {
//    sprintf (&text[str_len(text)], "M[%d] = ", i);
//    for (d = 0; d < multivector_array[i]->dimensionality; d ++)
//      sprintf (&text[str_len(text)], "[%d]",multivector_array[i]->dimension_size[d]);
//    sprintf (&text[str_len(text)], "(%d)",multivector_array[i]->data_type_size);
//    sprintf (&text[str_len(text)], "\n");
//  }
//  printf ("Multivector catalog:\n%s\n",text);
//}

//void * Multivector_2DAccess(Multivector * multivector, uint16_t row, uint16_t column)
//{
//  void * data = NULL;
//  ASSERT (multivector != NULL);
//  ASSERT (multivector->data != NULL);
//  ASSERT (2 <= multivector->dimensionality);
//  ASSERT (row <= multivector->dimension_size[0]);
//  ASSERT (column <= multivector->dimension_size[1]);
//
//  MultivectorArray_add(multivector);
//
//  if ((multivector != NULL)
//      && (multivector->data != NULL)
//      && (2 <= multivector->dimensionality)
//      && (row <= multivector->dimension_size[0])
//      && (column <= multivector->dimension_size[1]))
//  {
//    uint16_t dimensionality = multivector->dimensionality;
//    size_t data_size = multivector->data_type_size;
//
//    while (dimensionality-- > 2)
//    {
//      data_size *= multivector->dimension_size[dimensionality];
//    }
//
//    data = multivector->data
//        + (row * multivector->dimension_size[1] + column) * data_size;
//  }
//
//  return data;
//}
//
//void * Multivector_3DAccess (Multivector * multivector, uint16_t row, uint16_t column, uint16_t position)
//{
//  void * data = NULL;
//  ASSERT (multivector != NULL);
//  ASSERT (multivector->data != NULL);
//  ASSERT (3 <= multivector->dimensionality);
//  ASSERT (row <= multivector->dimension_size[0]);
//  ASSERT (column <= multivector->dimension_size[1]);
//  ASSERT (position <= multivector->dimension_size[2]);
//
//  MultivectorArray_add(multivector);
//
//  if ((multivector != NULL)
//      && (multivector->data != NULL)
//      && (3 <= multivector->dimensionality)
//      && (row <= multivector->dimension_size[0])
//      && (column <= multivector->dimension_size[1])
//      && (position <= multivector->dimension_size[2]))
//  {
//    uint16_t dimensionality = multivector->dimensionality;
//    size_t data_size = multivector->data_type_size;
//
//    while (dimensionality-- > 3)
//    {
//      data_size *= multivector->dimension_size[dimensionality];
//    }
//
//    data = multivector->data
//        + ((row * multivector->dimension_size[1] + column)
//            * multivector->dimension_size[2] + position) * data_size;
//  }
//
//  return data;
//}
void inline * Multivector_2DAccess (Multivector * multivector, uint16_t row, uint16_t column) __attribute__((always_inline));
void inline * Multivector_2DAccess (Multivector * multivector, uint16_t row, uint16_t column)
{
  ASSERT (multivector != NULL);
  ASSERT (multivector->data != NULL);
  ASSERT (2 <= multivector->dimensionality);
  ASSERT (row <= multivector->dimension_size[0]);
  ASSERT (column <= multivector->dimension_size[1]);

  switch (multivector->type_id)
  {
    case M32BIT_24_24_ID:
      return &(*(M32Bit_24_24*) multivector->data)[row][column];
    case M32BIT_24_24_50_ID:
      return &(*(M32Bit_24_24_50*) multivector->data)[row][column];
    case M32BIT_12_24_32_ID:
      return &(*(M32Bit_12_24_32*) multivector->data)[row][column];
    case M32BIT_1_1_50_32_ID:
      return &(*(M32Bit_1_1_50_32*) multivector->data)[row][column];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column];
    case M32BIT_12_12_ID:
      return &(*(M32Bit_12_12*) multivector->data)[row][column];
    case M32BIT_2_2_32_32_ID:
      return &(*(M32Bit_2_2_32_32*) multivector->data)[row][column];
    case M32BIT_8_8_64_ID:
      return &(*(M32Bit_8_8_64*) multivector->data)[row][column];
    case M32BIT_8_8_ID:
      return &(*(M32Bit_8_8*) multivector->data)[row][column];
    case M32BIT_5_5_32_64_ID:
      return &(*(M32Bit_5_5_32_64*) multivector->data)[row][column];
    case M32BIT_2_4_64_ID:
      return &(*(M32Bit_2_4_64*) multivector->data)[row][column];
    case M32BIT_4_4_ID:
      return &(*(M32Bit_4_4*) multivector->data)[row][column];
    case M32BIT_2_2_64_64_ID:
      return &(*(M32Bit_2_2_64_64*) multivector->data)[row][column];
    case M32BIT_1_1_1024_ID:
      return &(*(M32Bit_1_1_1024*) multivector->data)[row][column];
    case M32BIT_1_1_ID:
      return &(*(M32Bit_1_1*) multivector->data)[row][column];
    case M32BIT_4_4_64_1024_ID:
      return &(*(M32Bit_4_4_64_1024*) multivector->data)[row][column];
    case M32BIT_1_1_10_ID:
      return &(*(M32Bit_1_1_10*) multivector->data)[row][column];
    case M32BIT_1_1_1024_10_ID:
      return &(*(M32Bit_1_1_1024_10*) multivector->data)[row][column];
    default:
      ASSERT (0);
  }
  return NULL;
}

void inline * Multivector_3DAccess (Multivector * multivector, uint16_t row, uint16_t column, uint16_t position) __attribute__((always_inline));
void inline * Multivector_3DAccess (Multivector * multivector, uint16_t row, uint16_t column, uint16_t position)
{
  ASSERT(multivector != NULL);
  ASSERT(multivector->data != NULL);
  ASSERT(3 <= multivector->dimensionality);
  ASSERT(row <= multivector->dimension_size[0]);
  ASSERT(column <= multivector->dimension_size[1]);
  ASSERT(position <= multivector->dimension_size[2]);

  switch (multivector->type_id)
  {
    case M32BIT_24_24_50_ID:
      return &(*(M32Bit_24_24_50*) multivector->data)[row][column][position];
    case M32BIT_12_24_32_ID:
      return &(*(M32Bit_12_24_32*) multivector->data)[row][column][position];
    case M32BIT_1_1_50_32_ID:
      return &(*(M32Bit_1_1_50_32*) multivector->data)[row][column][position];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column][position];
    case M32BIT_2_2_32_32_ID:
      return &(*(M32Bit_2_2_32_32*) multivector->data)[row][column][position];
    case M32BIT_8_8_64_ID:
      return &(*(M32Bit_8_8_64*) multivector->data)[row][column][position];
    case M32BIT_5_5_32_64_ID:
      return &(*(M32Bit_5_5_32_64*) multivector->data)[row][column][position];
    case M32BIT_2_4_64_ID:
      return &(*(M32Bit_2_4_64*) multivector->data)[row][column][position];
    case M32BIT_2_2_64_64_ID:
      return &(*(M32Bit_2_2_64_64*) multivector->data)[row][column][position];
    case M32BIT_1_1_1024_ID:
      return &(*(M32Bit_1_1_1024*) multivector->data)[row][column][position];
    case M32BIT_4_4_64_1024_ID:
      return &(*(M32Bit_4_4_64_1024*) multivector->data)[row][column][position];
    case M32BIT_1_1_10_ID:
      return &(*(M32Bit_1_1_10*) multivector->data)[row][column][position];
    case M32BIT_1_1_1024_10_ID:
      return &(*(M32Bit_1_1_1024_10*) multivector->data)[row][column][position];
    default:
      ASSERT(0)
      ;
  }
  return NULL;
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

static void Multivector_cacheFlush(Multivector * multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    size_t data_size = multivector->data_type_size;
    int i;

    for (i = 0; i < multivector->dimensionality; i++)
      data_size *= multivector->dimension_size[i];

    Xil_DCacheFlushRange ((UINTPTR) multivector->data, data_size);
  }
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
void SbsAcceleratorProfie_initialize(SbsAcceleratorProfie * profile,
                                     Multivector * state_matrix,
                                     uint32_t kernel_size,
                                     float epsilon)
{
  ASSERT (profile != NULL);
  ASSERT (state_matrix != NULL);
  ASSERT (state_matrix->dimensionality == 3);
  ASSERT (state_matrix->data != NULL);
  ASSERT (0 < kernel_size);
  ASSERT (0.0 < epsilon);

  if ((profile != NULL)
      && (state_matrix != NULL)
      && (state_matrix->dimensionality == 3)
      && (state_matrix->data != NULL)
      && (0 < kernel_size)
      && (0.0 < epsilon))
  {
    profile->layerSize = state_matrix->dimension_size[0]
        * state_matrix->dimension_size[1];
    profile->vectorSize = state_matrix->dimension_size[2];
    profile->kernelSize = kernel_size * kernel_size;
    profile->epsilon = epsilon;
    profile->txBufferSize = profile->layerSize
                            * (profile->kernelSize + 1)
                            * profile->vectorSize
                            * state_matrix->data_type_size;

    profile->rxBuffer = state_matrix->data;
    profile->rxBufferSize = profile->layerSize
                            * profile->vectorSize
                            * state_matrix->data_type_size;

    profile->vectorBufferSize = profile->vectorSize * state_matrix->data_type_size;
  }
}

/*****************************************************************************/
static SbsLayerPartition * SbsLayerPartition_new (SbSUpdateAccelerator * accelerator,
                                                  uint16_t x_pos,
                                                  uint16_t y_pos,
                                                  uint16_t rows,
                                                  uint16_t columns,
                                                  uint16_t neurons)
{
  SbsLayerPartition * partition = NULL;

  partition = (SbsLayerPartition *) malloc (sizeof(SbsLayerPartition));

  ASSERT (partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = NULL;
    MemoryBlock * memory_def = NULL;

    memset (partition, 0x00, sizeof(SbsLayerPartition));

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

    partition->x_pos = x_pos;
    partition->y_pos = y_pos;
  }

  return partition;
}

static void SbsLayerPartition_delete(SbsLayerPartition ** partition)
{
  ASSERT (partition != NULL);
  ASSERT (*partition != NULL);

  if ((partition != NULL) && (*partition != NULL))
  {
    Multivector_delete (&((*partition)->state_matrix));
    if ((*partition)->weight_matrix != NULL)
      Multivector_delete (&((*partition)->weight_matrix));

    free (*partition);

    *partition = NULL;
  }
}

static void SbsLayerPartition_initializeIP (NeuronState * state_vector, uint16_t size)
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

static void SbsLayerPartition_initialize (SbsLayerPartition * partition,
                                          uint32_t kernel_size,
                                          float epsilon)
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
        SbsLayerPartition_initializeIP (&state_matrix_data[current_row_index + column * neurons], neurons);
      }
    }

    SbsAcceleratorProfie_initialize(&partition->profile,
                                    state_matrix,
                                    kernel_size, epsilon);
  }
}

static void SbsLayerPartition_cacheFlush (SbsLayerPartition * partition)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector_cacheFlush (partition->state_matrix);

    if (partition->weight_matrix != NULL)
      Multivector_cacheFlush (partition->weight_matrix);
  }
}

static void SbsLayerPartition_setWeights (SbsLayerPartition * partition,
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
                                   WeightShift weight_shift)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);

  if (layer != NULL)
  {
    int           i;
    Multivector * spike_matrix;
    SbSUpdateAccelerator * accelerator_group[NUM_ACCELERATOR_INSTANCES] = {0};

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    layer->layer_type = layer_type;

    switch (layer_type)
    {
      case CONVOLUTION_LAYER:
        if (neurons == 32)
        {
          layer->num_partitions = 2;//NUM_ACCELERATOR_INSTANCES;
          accelerator_group[0] = Accelerator_array[0];
          accelerator_group[1] = Accelerator_array[1];
        }
        else if (neurons == 64)
        {
          layer->num_partitions = 1;//NUM_ACCELERATOR_INSTANCES;
          accelerator_group[0] = Accelerator_array[2];
        }
        break;
      case POOLING_LAYER:
        layer->num_partitions = 2;//NUM_ACCELERATOR_INSTANCES;
        accelerator_group[0] = Accelerator_array[0];
        accelerator_group[1] = Accelerator_array[1];
        break;
      case FULLY_CONNECTED_LAYER:
        layer->num_partitions = 1;
        accelerator_group[0] = Accelerator_array[3];
        break;
      case OUTPUT_LAYER:
        layer->num_partitions = 1;
        accelerator_group[0] = Accelerator_array[0];
        break;
      case INPUT_LAYER:
        layer->num_partitions = 1;
        accelerator_group[0] = NULL;
        break;
      default:
        ASSERT (0);
    }

    layer->partition_array = (SbsLayerPartition **)
        malloc (layer->num_partitions * sizeof(SbsLayerPartition *));
    ASSERT(layer->partition_array != NULL);

    if (layer->partition_array != NULL)
    {
      uint16_t residual = (rows % layer->num_partitions);
      uint16_t rows_per_partition = ((rows - residual) / layer->num_partitions);
      uint16_t rows_current_partition;
      uint16_t pos_y = 0;
      uint16_t pos_x = 0;

      ASSERT (((rows - residual) % layer->num_partitions) == 0);

      SbSUpdateAccelerator * accelerator = NULL;
      for (i = 0; i < layer->num_partitions; i++)
      {
        accelerator = accelerator_group[i];

        if (0 < residual)
        {
          rows_current_partition = rows_per_partition + 1;
          residual--;
        }
        else
        {
          rows_current_partition = rows_per_partition;
        }

        layer->partition_array[i] = SbsLayerPartition_new (accelerator,
                                                           pos_x,
                                                           pos_y,
                                                           rows_current_partition,
                                                           columns,
                                                           neurons);

        pos_y += rows_current_partition;
      }
    }

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new (NULL, sizeof(SpikeID), 2, rows, columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    layer->spike_matrix = spike_matrix;

    /* Assign parameters */
    layer->rows    = rows;
    layer->columns = columns;
    layer->neurons = neurons;
    layer->kernel_size   = kernel_size;
    layer->kernel_stride = kernel_stride;
    layer->weight_shift  = weight_shift;
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
        SbsLayerPartition_delete (&((*layer)->partition_array[--(*layer)->num_partitions]));

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
      SbsLayerPartition_initialize (layer->partition_array[i],
                                    layer->kernel_size,
                                    layer->epsilon);
  }
}

static void SbsBaseLayer_cacheFlush(SbsBaseLayer * layer)
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
      SbsLayerPartition_cacheFlush(layer->partition_array[i]);
  }
}

static void SbsBaseLayer_giveWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if ((layer != NULL) && (((SbsBaseLayer *) layer)->partition_array != NULL))
  {
    int i;

    for (i = 0; i < ((SbsBaseLayer *) layer)->num_partitions; i++)
      SbsLayerPartition_setWeights (((SbsBaseLayer *) layer)->partition_array[i],
                                     weight_matrix);

    Multivector_delete ((Multivector**) &weight_matrix);
  }
}

static void SbsBaseLayer_setEpsilon(SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->epsilon = epsilon;
}

inline static SpikeID SbsStateVector_generateSpike (NeuronState * state_vector, uint16_t size) __attribute__((always_inline));

inline static SpikeID SbsStateVector_generateSpike (NeuronState * state_vector, uint16_t size)
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

      //ASSERT(sum <= 1 + 1e-5);

      if (random_s <= sum)
        return spikeID;
    }
  }

  return size - 1;
}

static void SbsLayerPartition_loadInput(SbsLayerPartition * partition, char * file_name, uint8_t * input_label)
{
  ASSERT(partition != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  if ((partition != NULL) && (file_name != NULL) && (input_label != NULL))
  {
#ifdef USE_XILINX
    FIL fil; /* File object */
    FRESULT rc;
    rc = f_open (&fil, file_name, FA_READ);
    ASSERT(rc == FR_OK);

    if (rc == FR_OK)
    {
      uint16_t       rows        = partition->state_matrix->dimension_size[0];
      uint16_t       columns     = partition->state_matrix->dimension_size[1];
      uint16_t       neurons     = partition->state_matrix->dimension_size[2];
      NeuronState *  data        = partition->state_matrix->data;

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
        rc = f_read (&fil, input_label, sizeof(uint8_t), &read_result);
        (*input_label)--;
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
      uint16_t rows = partition->state_matrix->dimension_size[0];
      uint16_t columns = partition->state_matrix->dimension_size[1];
      uint16_t neurons = partition->state_matrix->dimension_size[2];
      NeuronState * data = partition->state_matrix->data;

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
        read_result = fread(input_label, 1, sizeof(uint8_t), file);
        (*input_label) --;
        good_reading_flag = read_result == sizeof(uint8_t);
      }

      fclose(file);
      ASSERT(good_reading_flag);
    }
#endif
  }
}

static void SbsBaseLayer_loadInput (SbsBaseLayer * layer, char * file_name,
                                    uint8_t * input_label)
{
  ASSERT(layer != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  ASSERT(layer->layer_type == INPUT_LAYER);
  ASSERT(layer->num_partitions == 1);
  if ((layer != NULL) && (file_name != NULL) && (input_label != NULL))
  {
    SbsLayerPartition_loadInput (layer->partition_array[0], file_name,
                                 input_label);
  }
}

static void SbsBaseLayer_getOutputVector(SbsBaseLayer * layer,
                                         NeuronState ** output_vector,
                                         uint16_t * output_vector_size)
{
  ASSERT(layer != NULL);
  ASSERT(0 < layer->num_partitions);
  ASSERT(layer->layer_type == OUTPUT_LAYER);
  ASSERT(layer->partition_array[layer->num_partitions - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != NULL);

  if ((layer != NULL) && (0 < layer->num_partitions)
      && (layer->layer_type == OUTPUT_LAYER)
      && (layer->partition_array[layer->num_partitions - 1] != NULL)
      && (output_vector != NULL) && (output_vector_size != NULL))
  {
    SbsLayerPartition *  partition = layer->partition_array[layer->num_partitions - 1];
    Multivector * output_state_matrix = partition->state_matrix;

    ASSERT(output_state_matrix->data != NULL);
    ASSERT(output_state_matrix->dimensionality == 3);
    ASSERT(output_state_matrix->dimension_size[0] == 1);
    ASSERT(output_state_matrix->dimension_size[1] == 1);
    ASSERT(0 < output_state_matrix->dimension_size[2]);

    * output_vector = output_state_matrix->data;
    * output_vector_size = output_state_matrix->dimension_size[2];
  }
}

inline SbsLayerPartition * SbsBaseLayer_getPartition(SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                              uint16_t * partition_row, uint16_t * partition_column) __attribute__((always_inline));

inline SbsLayerPartition * SbsBaseLayer_getPartition(SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                              uint16_t * partition_row, uint16_t * partition_column)
{
  SbsLayerPartition * partition = NULL;
  if (layer->num_partitions == 1)
  {
    partition = layer->partition_array[0];
    if (partition_row) *partition_row = row;
    if (partition_column) *partition_column = column;
  }
  else
  {
    int i;
    for (i = 0; partition == NULL && i < (layer)->num_partitions; i++)
    {
      if (layer->partition_array[i]->x_pos <= column
          && column < layer->partition_array[i]->x_pos + layer->partition_array[i]->state_matrix->dimension_size[1]
          && layer->partition_array[i]->y_pos <= row
          && row < layer->partition_array[i]->y_pos + layer->partition_array[i]->state_matrix->dimension_size[0])
      {
        partition = layer->partition_array[i];
        if (partition_row) *partition_row = row - layer->partition_array[i]->y_pos;
        if (partition_column) *partition_column = column - layer->partition_array[i]->x_pos;
      }
    }
  }

  return partition;
}

static void SbsBaseLayer_generateSpikes (SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);
  ASSERT(layer->spike_matrix != NULL);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions)
      && (layer->spike_matrix != NULL))
  {
    int i;
    uint16_t columns = layer->columns;
    uint16_t neurons = layer->neurons;

    uint16_t partition_row = 0;
    SbsLayerPartition *  partition = NULL;
    Multivector * partition_state_matrix = NULL;
    Multivector* layer_spike_matrix = layer->spike_matrix;

    SpikeID * spike;
    NeuronState * state_vector;

    uint16_t row;
    uint16_t column;

    for (i = 0; i < layer->num_partitions; i ++)
    {
      ASSERT(layer->partition_array[i] != NULL);
      ASSERT(layer->partition_array[i]->state_matrix != NULL);
      partition = layer->partition_array[i];
      partition_state_matrix = partition->state_matrix;

      for (row = partition->y_pos, partition_row = 0;
          partition_row < partition_state_matrix->dimension_size[0];
          partition_row++, row ++)
      {
        for (column = 0; column < columns; column++)
        {
          spike = Multivector_2DAccess (layer_spike_matrix, row, column);
          state_vector = Multivector_2DAccess (partition_state_matrix,
                                               partition_row,
                                               column);
          *spike = SbsStateVector_generateSpike (state_vector, neurons);
        }
      }
    }
  }
}

inline static void SbsBaseLayer_update(SbsBaseLayer * layer, SbsBaseLayer * spike_layer) __attribute__((always_inline));
inline static void SbsBaseLayer_update(SbsBaseLayer * layer, SbsBaseLayer * spike_layer)
{
  ASSERT (layer != NULL);
  ASSERT (spike_layer != NULL);
  if ((layer != NULL) && (spike_layer != NULL))
  {
    int i;
    SbsLayerPartition *  update_partition = NULL;
    uint16_t             update_partition_row;

    SpikeID   spikeID       = 0;
    Multivector * spike_layer_spike_matrix = spike_layer->spike_matrix;


    Weight * weight_vector  = NULL;
    NeuronState* state_vector;


    uint16_t kernel_stride  = layer->kernel_stride;
    uint16_t kernel_size    = layer->kernel_size;


    uint16_t layer_row;         /* Row index for navigation on the layer */
    uint16_t layer_column;      /* Column index for navigation on the layer */
    uint16_t kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
    uint16_t kernel_row_pos;    /* Kernel row position for navigation on the spike matrix */
    uint16_t kernel_row;        /* Row index for navigation inside kernel */
    uint16_t kernel_column;     /* Column index for navigation inside kernel */

    uint16_t layer_columns = layer->columns;

    Multivector * update_partition_weight_matrix = NULL;
    SbSUpdateAccelerator * update_partition_accelerator = NULL;
    Multivector * update_partition_state_matrix = NULL;

    Multivector * layer_spike_matrix = layer->spike_matrix;
    SpikeID * spike_matrix_data = NULL;

    WeightShift layer_weight_shift = layer->weight_shift;

    uint16_t layer_neurons = layer->neurons;

    kernel_row_pos = 0, layer_row = 0;
    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      update_partition_weight_matrix = update_partition->weight_matrix;
      update_partition_accelerator = update_partition->accelerator;
      update_partition_state_matrix = update_partition->state_matrix;

      Accelerator_setup (update_partition_accelerator,
                         &update_partition->profile);

      /* Update begins */
      for (update_partition_row = 0;
          update_partition_row < update_partition_state_matrix->dimension_size[0];
           update_partition_row ++,
           kernel_row_pos += kernel_stride, layer_row ++)
      {
        for (kernel_column_pos = 0, layer_column = 0;
            layer_column < layer_columns;
             kernel_column_pos += kernel_stride, layer_column ++)
        {
          state_vector = Multivector_2DAccess(update_partition_state_matrix, update_partition_row, layer_column);

          spike_matrix_data = Multivector_2DAccess(layer_spike_matrix, layer_row, layer_column);

          * spike_matrix_data = SbsStateVector_generateSpike (state_vector, layer_neurons);

          Accelerator_giveStateVector (update_partition_accelerator, state_vector);

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              spikeID = *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

              ASSERT(layer->neurons == update_partition->weight_matrix->dimension_size[3]);

              if (layer_weight_shift == COLUMN_SHIFT)
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_row, kernel_column, spikeID);
              }
              else
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_column, kernel_row, spikeID);
              }

              Accelerator_giveWeightVector (update_partition_accelerator, weight_vector);
            }
          }
        }
      }
      /* Update ends */
      Accelerator_start (update_partition_accelerator);
    }
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
    SbsBaseLayer_loadInput (network->layer_array[0], file_name,
                            &network->input_label);
  }
}

static void SbsBaseNetwork_updateCycle(SbsNetwork * network_ptr, uint16_t cycles)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  Timer * timer = Timer_new (1);
  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    int i;
    SbsBaseLayer_cacheFlush(network->layer_array[0]);
    /* Initialize all layers except the input-layer */
    for (i = 1; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
      SbsBaseLayer_cacheFlush(network->layer_array[i]);
    }

    Timer_start(timer);
    /************************ Begins Update cycle **************************/
    while (cycles--)
    {
      for (i = 0; i <= network->size - 1; i ++)
      {
//        if (i < network->size - 1)
        if (i == 0)
          SbsBaseLayer_generateSpikes (network->layer_array[i]);

        layer_wait = i;
        if (0 < i) SbsBaseLayer_update (network->layer_array[i],
                                        network->layer_array[i - 1]);
      }

      if (cycles % 100 == 0)
        printf ("%d %f S\n", cycles, Timer_getCurrentTime (timer));
    }
    /************************ Ends Update cycle ****************************/

    /************************ Get inferred output **************************/
    {
      NeuronState max_value = 0;
      SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
      NeuronState * output_state_vector = NULL;
      uint16_t output_vector_size = 0;

      SbsBaseLayer_getOutputVector (output_layer, &output_state_vector,
                                    &output_vector_size);

      ASSERT(output_state_vector != NULL);
      ASSERT(0 < output_vector_size);

      for (i = 0; i < output_vector_size; i++)
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

static void SbsBaseNetwork_getOutputVector(SbsNetwork * network_ptr,
                                           NeuronState ** output_vector,
                                           uint16_t * output_vector_size)
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
      && (network->layer_array[network->size - 1] != NULL)
      && (output_vector != NULL)
      && (output_vector_size != NULL))
  {
    SbsBaseLayer_getOutputVector (network->layer_array[network->size - 1],
                                  output_vector, output_vector_size);
  }
}

static size_t SbsBaseNetwork_getMemorySize (SbsNetwork * network)
{
  for (int l = 0; l < ((SbsBaseNetwork *) network)->size; l++)
    for (int a = 0; a < 4; a++)
    {
      if (accelerator_wait[l][a]) printf ("accelerator_wait[%d][%d] = %d\n", l, a,
                                          accelerator_wait[l][a]);

      if (tx_wait[l][a]) printf ("tx_wait[%d][%d] = %d\n", l, a, tx_wait[l][a]);

      if (rx_wait[l][a]) printf ("rx_wait[%d][%d] = %d\n", l, a, rx_wait[l][a]);
    }

  //MultivectorArray_print ();
  return 0;
}
/*****************************************************************************/

static SbsLayer * SbsInputLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new (INPUT_LAYER, rows, columns, neurons, 0,
                                        0, ROW_SHIFT);
}

static SbsLayer * SbsConvolutionLayer_new(uint16_t rows,
                                            uint16_t columns,
                                            uint16_t neurons,
                                            uint16_t kernel_size,
                                            WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (CONVOLUTION_LAYER, rows, columns,
                                        neurons, kernel_size, 1, weight_shift);
}

static SbsLayer * SbsPoolingLayer_new(uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons,
                                    uint16_t kernel_size,
                                    WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (POOLING_LAYER, rows, columns, neurons,
                                        kernel_size, kernel_size, weight_shift);
}

static SbsLayer * SbsFullyConnectedLayer_new(uint16_t neurons,
                                                  uint16_t kernel_size,
                                                  WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (FULLY_CONNECTED_LAYER, 1, 1, neurons,
                                        kernel_size, 1, weight_shift);
}

static SbsLayer * SbsOutputLayer_new(uint16_t neurons,
                                     WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (OUTPUT_LAYER, 1, 1, neurons, 1, 1,
                                        weight_shift);
}
/*****************************************************************************/

static SbsWeightMatrix SbsWeightMatrix_new (uint16_t rows,
                                            uint16_t columns,
                                            uint16_t depth,
                                            uint16_t neurons,
                                            char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new(NULL, sizeof(Weight), 4, rows, columns, depth, neurons);

    ASSERT(weight_watrix != NULL);
    ASSERT(weight_watrix->dimensionality == 4);
    ASSERT(weight_watrix->data != NULL);
    ASSERT(weight_watrix->dimension_size[0] == rows);
    ASSERT(weight_watrix->dimension_size[1] == columns);
    ASSERT(weight_watrix->dimension_size[2] == depth);
    ASSERT(weight_watrix->dimension_size[3] == neurons);

    if ((weight_watrix != NULL)
        && (weight_watrix->dimensionality == 4)
        && (weight_watrix->data != NULL)
        && (weight_watrix->dimension_size[0] == rows)
        && (weight_watrix->dimension_size[1] == columns)
        && (weight_watrix->dimension_size[2] == depth)
        && (weight_watrix->dimension_size[3] == neurons))
    {
#ifdef USE_XILINX
      FIL fil; /* File object */
      FRESULT rc;
      rc = f_open (&fil, file_name, FA_READ);
      ASSERT(rc == FR_OK);

      if (rc == FR_OK)
      {
        size_t read_size;
        size_t data_size = rows * columns * depth * neurons * sizeof(Weight);
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
        size_t data_size = rows * columns * depth * neurons * sizeof(Weight);
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
