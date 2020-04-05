/*
 * sbs_hardware_emulator.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_hardware_emulator.h"
#include "stdio.h"
#include "stdlib.h"

#include "miscellaneous.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

typedef enum
{
  HW_START    = 1 << 0,  //        bit 0  - ap_start (Read/Write/COH)
  HW_DONE     = 1 << 1,  //        bit 1  - ap_done (Read/COR)
  HW_IDLE     = 1 << 2,  //        bit 2  - ap_idle (Read)
  HW_READY    = 1 << 3,  //        bit 3  - ap_ready (Read)
  HW_RESTART  = 1 << 7   //        bit 7  - auto_restart (Read/Write)
} HWCtrlFlags;

typedef struct
{
  uint32_t    deviceId;
  DMAIRQMask  rxInterruptMask;
  DMAIRQMask  txInterruptMask;
  uint32_t    interruptVectorId;
  ARM_GIC_InterruptHandler interruptHandler;

  void * txBufferAddres;
  uint32_t txBufferLength;
  uint32_t txIndex;

  void * rxBufferAddres;
  uint32_t rxBufferLength;
  uint32_t rxIndex;

  void *      interruptContextData;
} DMAHwEmulator;

typedef struct
{
  uint32_t  deviceId;
  uint8_t   interruptGlobalEnable :1;
  uint32_t  interruptMask;
  uint32_t  interruptVectorId;
  ARM_GIC_InterruptHandler interruptHandler;
  void *      interruptContextData;

  uint32_t debug;
  uint32_t layerSize;
  uint32_t kernelSize;
  uint32_t vectorSize;
  uint32_t epsilon;

  uint32_t hwCtrlFlags;
} SbsHwUpdateEmulator;

typedef struct
{
  SbsHwUpdateEmulator hwUpdate;
  uint32_t            hwSpike;
  DMAHwEmulator       hwDMA;
} SbsHardwareEmulator;

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/
SbsHardwareEmulator SbsHardwareEmulator_instance;
/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/
#define H_QF    (21)
#define W_QF    (16)
#define H_MAX   (((unsigned long)1 << H_QF) - 1)
#define W_MAX   (((unsigned long)1 << W_QF) - 1)

#define EPSILON_DIV_SUM_EX_QF (H_QF) // From 0 to H_QF
#define REV_DIV_EPSILON_EX_QF (H_QF)


typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

#define MAX_VECTOR_SIZE       (1024)
#define MAX_SPIKE_MATRIX_SIZE (60*60)


typedef struct
{
  uint32_t data;
  uint32_t last;
} StreamChannel;

typedef struct
{
  StreamChannel (* read)(void);
  void (* write)(StreamChannel);
} Stream;

StreamChannel read(void)
{
  StreamChannel channel;
  channel.data = ((uint32_t*)SbsHardwareEmulator_instance.hwDMA.txBufferAddres)[SbsHardwareEmulator_instance.hwDMA.txIndex];
  SbsHardwareEmulator_instance.hwDMA.txIndex ++;
  return channel;
}

void write(StreamChannel channel)
{
  ((uint32_t*)SbsHardwareEmulator_instance.hwDMA.rxBufferAddres)[SbsHardwareEmulator_instance.hwDMA.rxIndex] = channel.data;
  SbsHardwareEmulator_instance.hwDMA.rxIndex ++;
}

uint64_t wide_div(uint64_t dividend, uint64_t divisor)
{
  return dividend / divisor;
}

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0x0000FFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (0x000000FF & (((unsigned int)(d)) >> 19))

#define NEGLECTING_CONSTANT   ((float)1e-20)

static int debug_flags;

void sbs_accelerator (Stream stream_in,
                      Stream stream_out,
                      int * debug,
                      unsigned int layerSize,
                      unsigned int kernelSize,
                      unsigned int vectorSize,
                      unsigned int epsilon)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=kernelSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=epsilon     bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static unsigned int index;
  static unsigned int index_channel;

  static int ip_index;
  static unsigned short spikeID;
  static int i;
  static int batch;
  static StreamChannel channel;

  static unsigned short spike_matrix[MAX_SPIKE_MATRIX_SIZE];
#pragma HLS array_partition variable=spike_matrix block factor=4

  static int spike_index;

  static float state_vector[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=state_vector block factor=246

  static float weight_vector[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=weight_vector block factor=246

  static float temp_data[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=temp_data block factor=246

  static float epsion_over_sum;
  static float random_value;

  static unsigned int temp;

  static Data32 register_A;
  static Data32 register_B;

  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  static float sum;

  temp = 0;
  index_channel = 0;
  debug_flags = 1;

  for (ip_index = 0; ip_index < layerSize; ip_index++)
  {
#pragma HLS pipeline
    if (ip_index == 0)
    {
      channel = stream_in.read ();
      register_A.u32 = channel.data;
    }
    else
    {
      register_A.u32 = stream_in.read ().data;
    }
    random_value = register_A.f32;

    index = 0;
    while (index < vectorSize)
    {
#pragma HLS pipeline
      i = stream_in.read ().data;

      register_B.u32 = DATA16_TO_FLOAT32(i >> 0);
      state_vector[index] = register_B.f32;
      index ++;

      if (index < vectorSize)
      {
        register_B.u32 = DATA16_TO_FLOAT32(i >> 16);
        state_vector[index] = register_B.f32;
        index++;
      }
    }

    sum = 0.0f;
    for (spikeID = 0; spikeID < vectorSize; spikeID++)
    {
#pragma HLS pipeline
      if (sum < random_value)
      {
        sum += state_vector[spikeID];

        if (random_value <= sum || (spikeID == vectorSize - 1))
        {
          spike_matrix[ip_index] = spikeID;
        }
      }
    }

    for (batch = 0; batch < kernelSize; batch++)
    {
#pragma HLS pipeline
      index = 0;
      while (index < vectorSize)
      {
  #pragma HLS pipeline
        i = stream_in.read ().data;

        register_B.u32 = DATA8_TO_FLOAT32(i >> 0);
        weight_vector[index] = register_B.f32;
        index++;

        if (index < vectorSize)
        {
          register_B.u32 = DATA8_TO_FLOAT32(i >> 8);
          weight_vector[index] = register_B.f32;
          index++;
        }

        if (index < vectorSize)
        {
          register_B.u32 = DATA8_TO_FLOAT32(i >> 16);
          weight_vector[index] = register_B.f32;
          index++;
        }

        if (index < vectorSize)
        {
          register_B.u32 = DATA8_TO_FLOAT32(i >> 24);
          weight_vector[index] = register_B.f32;
          index++;
        }
      }

      sum = 0.0f;
      for (i = 0; i < vectorSize; i++)
      {
#pragma HLS pipeline
        temp_data[i] = state_vector[i] * weight_vector[i];
        sum += temp_data[i];
      }

      if (NEGLECTING_CONSTANT < sum)
      {
        epsion_over_sum = epsilon / sum;
        for (i = 0; i < vectorSize; i++)
        {
#pragma HLS pipeline
          state_vector[i] = reverse_epsilon
              * (state_vector[i] + temp_data[i] * epsion_over_sum);
        }
      }
    }


    for (i = 0; i < vectorSize; i++)
    {
#pragma HLS pipeline
      register_A.f32 = state_vector[i];

      if ((register_A.u32 & 0xf0000000) == 0x30000000)
      {
        temp |= (FLOAT32_TO_DATA16(register_A.u32)) << (16 * index_channel);
      }

      index_channel ++;

      if (index_channel == 2)
      {
        channel.data = temp;
        stream_out.write (channel);
        index_channel = 0;
        temp = 0;
      }
    }
  }

  index_channel = 0;
  temp = 0;
  for (i = 0; i < layerSize; i++)
  {
#pragma HLS pipeline
    temp |= ((unsigned int)spike_matrix[i]) << (16 * index_channel);
    index_channel ++;

    if ((index_channel == 2) || (i == layerSize - 1))
    {
      channel.data = temp;
      channel.last = (i == layerSize - 1);
      stream_out.write (channel);
      index_channel = 0;
      temp = 0;
    }
  }
}

void SbsHardwareEmulator_trigger (SbsHardwareEmulator * instance)
{
  ASSERT (instance != NULL);
  if (instance != NULL)
  {
    if ((instance->hwUpdate.hwCtrlFlags & HW_START)
        && (instance->hwDMA.rxBufferAddres != NULL)
        && (instance->hwDMA.txBufferAddres != NULL))
    {
      Stream stream_in = {read, write};
      Stream stream_out = {read, write};
      sbs_accelerator (stream_in,
                       stream_out,
                       &instance->hwUpdate.debug,
                       instance->hwUpdate.layerSize,
                       instance->hwUpdate.kernelSize,
                       instance->hwUpdate.vectorSize,
                       instance->hwUpdate.epsilon);

      instance->hwDMA.rxBufferAddres = NULL;
      instance->hwDMA.txBufferAddres = NULL;

      if (instance->hwDMA.interruptHandler)
        instance->hwDMA.interruptHandler (instance->hwDMA.interruptContextData);

      if (instance->hwUpdate.interruptHandler)
        instance->hwUpdate.interruptHandler (instance->hwUpdate.interruptContextData);
    }
  }
}
/*****************************************************************************/

static void * SbsHwUpdateEmulator_new(void)
{
  return (void *) &SbsHardwareEmulator_instance.hwUpdate;
}

static void SbsHwUpdateEmulator_delete (void ** InstancePtr)
{
//  if (InstancePtr && *InstancePtr)
//  {
//    free (*InstancePtr);
//    *InstancePtr = NULL;
//  }
}

static int SbsHwUpdateEmulator_Initialize(void * instance, u16 deviceId)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    memset (instance, 0x00, sizeof(SbsHwUpdateEmulator));

    ((SbsHwUpdateEmulator *) instance)->deviceId = deviceId;

    ((SbsHwUpdateEmulator *) instance)->hwCtrlFlags |= HW_IDLE | HW_READY;
  }
  else return XST_FAILURE;

  return XST_SUCCESS;
}

static void SbsHwUpdateEmulator_Start (void * instance)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    if (((SbsHwUpdateEmulator*) instance)->hwCtrlFlags & HW_READY)
    {
      ((SbsHwUpdateEmulator*) instance)->hwCtrlFlags &= ~HW_DONE;
      ((SbsHwUpdateEmulator*) instance)->hwCtrlFlags |= HW_START;
    }
    SbsHardwareEmulator_trigger (&SbsHardwareEmulator_instance);
  }
}

static uint32_t SbsHwUpdateEmulator_IsDone (void * instance)
{
  uint32_t isDone = 0;
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    isDone = ((SbsHwUpdateEmulator*) instance)->hwCtrlFlags & HW_DONE;
  }

  return isDone;
}

static uint32_t SbsHwUpdateEmulator_IsIdle (void * instance)
{
  uint32_t isIdle = 0;
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    isIdle = ((SbsHwUpdateEmulator*) instance)->hwCtrlFlags & HW_IDLE;
  }

  return isIdle;
}

static uint32_t SbsHwUpdateEmulator_IsReady (void * instance)
{
  uint32_t isRedy = 0;
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    isRedy = ((SbsHwUpdateEmulator*) instance)->hwCtrlFlags & HW_READY;
  }

  return isRedy;
}

static void SbsHwUpdateEmulator_EnableAutoRestart (void * instance)
{

}

static void SbsHwUpdateEmulator_DisableAutoRestart (void * instance)
{

}

static void SbsHwUpdateEmulator_Set_layerSize (void * instance, uint32_t layerSize)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator*) instance)->layerSize = layerSize;
  }
}

static uint32_t SbsHwUpdateEmulator_Get_layerSize (void * instance)
{
  ASSERT (instance != NULL);
  uint32_t layerSize = 0;

  if (instance != NULL)
  {
    layerSize = ((SbsHwUpdateEmulator*) instance)->layerSize;
  }

  return layerSize;
}

static void SbsHwUpdateEmulator_Set_kernelSize (void * instance, uint32_t kernelSize)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator*) instance)->kernelSize = kernelSize;
  }
}

static uint32_t SbsHwUpdateEmulator_Get_kernelSize (void * instance)
{
  ASSERT (instance != NULL);
  uint32_t kernelSize = 0;

  if (instance != NULL)
  {
    kernelSize = ((SbsHwUpdateEmulator*) instance)->kernelSize;
  }

  return kernelSize;
}

static void SbsHwUpdateEmulator_Set_vectorSize (void * instance, uint32_t vectorSize)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator*) instance)->vectorSize = vectorSize;
  }
}

static uint32_t SbsHwUpdateEmulator_Get_vectorSize (void * instance)
{
  ASSERT (instance != NULL);
  uint32_t vectorSize = 0;

  if (instance != NULL)
  {
    vectorSize = ((SbsHwUpdateEmulator*) instance)->vectorSize;
  }

  return vectorSize;
}

static void SbsHwUpdateEmulator_Set_epsilon (void * instance, uint32_t epsilon)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator*) instance)->epsilon = epsilon;
  }
}

static uint32_t SbsHwUpdateEmulator_Get_epsilon (void * instance)
{
  ASSERT (instance != NULL);
  uint32_t epsilon = 0;

  if (instance != NULL)
  {
    epsilon = ((SbsHwUpdateEmulator*) instance)->epsilon;
  }

  return epsilon;
}

static void SbsHwUpdateEmulator_InterruptGlobalEnable(void * instance)
{
  ASSERT(instance != NULL);

  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator *) instance)->interruptGlobalEnable = 1;
  }
}

static void SbsHwUpdateEmulator_InterruptGlobalDisable (void * instance)
{
  ASSERT(instance != NULL);

  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator *) instance)->interruptGlobalEnable = 0;
  }
}

static void SbsHwUpdateEmulator_InterruptEnable (void * instance, uint32_t mask)
{
  ASSERT(instance != NULL);
  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator*) instance)->interruptMask |= mask;
  }
}

static void SbsHwUpdateEmulator_InterruptDisable (void * instance, uint32_t mask)
{
  ASSERT(instance != NULL);
  if (instance != NULL)
  {
    ((SbsHwUpdateEmulator*) instance)->interruptMask &= ~mask;
  }
}

static void SbsHwUpdateEmulator_InterruptClear (void * instance, uint32_t mask)
{

}

static uint32_t SbsHwUpdateEmulator_InterruptGetEnabled (void * instance)
{

  return XST_SUCCESS;
}

static uint32_t SbsHwUpdateEmulator_InterruptGetStatus (void * instance)
{

  return 1;
}

static uint32_t  SbsHwUpdateEmulator_InterruptSetHandler (void *instance,
                                                          uint32_t interruptId,
                                                          ARM_GIC_InterruptHandler handler,
                                                          void * data)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    SbsHwUpdateEmulator * update_emulator = (SbsHwUpdateEmulator *) instance;
    update_emulator->interruptVectorId = interruptId;
    update_emulator->interruptHandler = handler;
    update_emulator->interruptContextData = data;
  }
  else return XST_FAILURE;

  return XST_SUCCESS;
}

SbsHardware SbsHardware_HwUpdateEmulator =
{
  .new =                SbsHwUpdateEmulator_new,
  .delete =             SbsHwUpdateEmulator_delete,

  .Initialize =         SbsHwUpdateEmulator_Initialize,
  .Start =              SbsHwUpdateEmulator_Start,
  .IsDone =             SbsHwUpdateEmulator_IsDone,
  .IsIdle =             SbsHwUpdateEmulator_IsIdle,
  .IsReady =            SbsHwUpdateEmulator_IsReady,
  .EnableAutoRestart =  SbsHwUpdateEmulator_EnableAutoRestart,
  .DisableAutoRestart = SbsHwUpdateEmulator_DisableAutoRestart,

  .Set_mode =           NULL,
  .Get_mode =           NULL,
  .Set_layerSize =      SbsHwUpdateEmulator_Set_layerSize,
  .Get_layerSize =      SbsHwUpdateEmulator_Get_layerSize,
  .Set_kernelSize =     SbsHwUpdateEmulator_Set_kernelSize,
  .Get_kernelSize =     SbsHwUpdateEmulator_Get_kernelSize,
  .Set_vectorSize =     SbsHwUpdateEmulator_Set_vectorSize,
  .Get_vectorSize =     SbsHwUpdateEmulator_Get_vectorSize,
  .Set_epsilon =        SbsHwUpdateEmulator_Set_epsilon,
  .Get_epsilon =        SbsHwUpdateEmulator_Get_epsilon,
  .Set_debug =          NULL,
  .Get_debug =          NULL,

  .InterruptGlobalEnable =  SbsHwUpdateEmulator_InterruptGlobalEnable,
  .InterruptGlobalDisable = SbsHwUpdateEmulator_InterruptGlobalDisable,
  .InterruptEnable =        SbsHwUpdateEmulator_InterruptEnable,
  .InterruptDisable =       SbsHwUpdateEmulator_InterruptDisable,
  .InterruptClear =         SbsHwUpdateEmulator_InterruptClear,
  .InterruptGetEnabled =    SbsHwUpdateEmulator_InterruptGetEnabled,
  .InterruptGetStatus =     SbsHwUpdateEmulator_InterruptGetStatus,

  .InterruptSetHandler =    SbsHwUpdateEmulator_InterruptSetHandler
};

///////////////////////////////////////////////////////////////////////////////

static void * SbsHwSpikeEmulator_new (void)
{
// return malloc (sizeof(int));
  return (void * )&SbsHardwareEmulator_instance.hwSpike;
}

static void SbsHwSpikeEmulator_delete (void ** InstancePtr)
{
//  if (InstancePtr && *InstancePtr)
//  {
//    free (*InstancePtr);
//    *InstancePtr = NULL;
//  }
}

static int SbsHwSpikeEmulator_Initialize(void * instance, u16 ID)
{

  return XST_SUCCESS;
}

static void SbsHwSpikeEmulator_Start (void * instance)
{

}

static uint32_t SbsHwSpikeEmulator_IsDone (void * instance)
{

  return XST_SUCCESS;
}

static uint32_t SbsHwSpikeEmulator_IsIdle (void * instance)
{

  return XST_SUCCESS;
}

static uint32_t SbsHwSpikeEmulator_IsReady (void * instance)
{

  return XST_SUCCESS;
}

static void SbsHwSpikeEmulator_EnableAutoRestart (void * instance)
{

}

static void SbsHwSpikeEmulator_DisableAutoRestart (void * instance)
{

}

static void SbsHwSpikeEmulator_Set_layerSize (void * instance, uint32_t layer_size)
{

}

static uint32_t SbsHwSpikeEmulator_Get_layerSize (void * instance)
{

  return XST_SUCCESS;
}

static void SbsHwSpikeEmulator_Set_vectorSize (void * instance, uint32_t vector_size)
{

}

static uint32_t SbsHwSpikeEmulator_Get_vectorSize (void * instance)
{

  return XST_SUCCESS;
}

static void SbsHwSpikeEmulator_InterruptGlobalEnable (void * instance)
{

}

static void SbsHwSpikeEmulator_InterruptGlobalDisable (void * instance)
{

}

static void SbsHwSpikeEmulator_InterruptEnable (void * instance, uint32_t mask)
{

}

static void SbsHwSpikeEmulator_InterruptDisable (void * instance, uint32_t mask)
{

}

static void SbsHwSpikeEmulator_InterruptClear (void * instance, uint32_t mask)
{

}

static uint32_t SbsHwSpikeEmulator_InterruptGetEnabled (void * instance)
{

  return XST_SUCCESS;
}

static uint32_t SbsHwSpikeEmulator_InterruptGetStatus (void * instance)
{

  return XST_SUCCESS;
}

static uint32_t  SbsHwSpikeEmulator_InterruptSetHandler (void *instance,
                                                         uint32_t ID,
                                                         ARM_GIC_InterruptHandler handler,
                                                         void * data)
{

  return XST_SUCCESS;
}

SbsHardware SbsHardware_HwSpikeEmulator =
{
  .new =                SbsHwSpikeEmulator_new,
  .delete =             SbsHwSpikeEmulator_delete,

  .Initialize =         SbsHwSpikeEmulator_Initialize,
  .Start =              SbsHwSpikeEmulator_Start,
  .IsDone =             SbsHwSpikeEmulator_IsDone,
  .IsIdle =             SbsHwSpikeEmulator_IsIdle,
  .IsReady =            SbsHwSpikeEmulator_IsReady,
  .EnableAutoRestart =  SbsHwSpikeEmulator_EnableAutoRestart,
  .DisableAutoRestart = SbsHwSpikeEmulator_DisableAutoRestart,

  .Set_mode =           NULL,
  .Get_mode =           NULL,
  .Set_layerSize =      SbsHwSpikeEmulator_Set_layerSize,
  .Get_layerSize =      SbsHwSpikeEmulator_Get_layerSize,
  .Set_kernelSize =     NULL,
  .Get_kernelSize =     NULL,
  .Set_vectorSize =     SbsHwSpikeEmulator_Set_vectorSize,
  .Get_vectorSize =     SbsHwSpikeEmulator_Get_vectorSize,
  .Set_epsilon =        NULL,
  .Get_epsilon =        NULL,
  .Set_debug =          NULL,
  .Get_debug =          NULL,

  .InterruptGlobalEnable =  SbsHwSpikeEmulator_InterruptGlobalEnable,
  .InterruptGlobalDisable = SbsHwSpikeEmulator_InterruptGlobalDisable,
  .InterruptEnable =        SbsHwSpikeEmulator_InterruptEnable,
  .InterruptDisable =       SbsHwSpikeEmulator_InterruptDisable,
  .InterruptClear =         SbsHwSpikeEmulator_InterruptClear,
  .InterruptGetEnabled =    SbsHwSpikeEmulator_InterruptGetEnabled,
  .InterruptGetStatus =     SbsHwSpikeEmulator_InterruptGetStatus,

  .InterruptSetHandler =    SbsHwSpikeEmulator_InterruptSetHandler
};

///////////////////////////////////////////////////////////////////////////////


static void * DMAHwEmulator_new(void)
{
  return (void *) &SbsHardwareEmulator_instance.hwDMA;
}

static void DMAHwEmulator_delete (void ** InstancePtr)
{
//  if (InstancePtr && *InstancePtr)
//  {
//    free (*InstancePtr);
//    *InstancePtr = NULL;
//  }
}

static int DMAHwEmulator_Initialize (void * instance, uint16_t deviceId)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    memset (instance, 0x00, sizeof(DMAHwEmulator));

    ((DMAHwEmulator *) instance)->deviceId = deviceId;
  }
  else return XST_FAILURE;

  return XST_SUCCESS;
}

static uint32_t  DMAHwEmulator_Move (void * instance,
                                     void * bufferAddres,
                                     uint32_t bufferLength,
                                     DMATransferDirection direction)
{
  ASSERT(instance != NULL);
  if (instance != NULL)
  {
    DMAHwEmulator * dma_emulator = (DMAHwEmulator *) instance;
    switch (direction)
    {
      case MEMORY_TO_HARDWARE:
        dma_emulator->txBufferAddres = bufferAddres;
        dma_emulator->txBufferLength = bufferLength;
        dma_emulator->txIndex = 0;
        break;
      case HARDWARE_TO_MEMORY:
        dma_emulator->rxBufferAddres = bufferAddres;
        dma_emulator->rxBufferLength = bufferLength;
        dma_emulator->rxIndex = 0;
        break;
      default: ASSERT(NULL);
    }

    SbsHardwareEmulator_trigger (&SbsHardwareEmulator_instance);
  }
  return XST_SUCCESS;
}

static void DMAHwEmulator_InterruptEnable (void * instance,
                                           DMAIRQMask mask,
                                           DMATransferDirection direction)
{
  ASSERT(instance != NULL);
  if (instance != NULL)
  {
    switch (direction)
    {
      case MEMORY_TO_HARDWARE:
        ((DMAHwEmulator *) instance)->txInterruptMask |= mask;
        break;
      case HARDWARE_TO_MEMORY:
        ((DMAHwEmulator *) instance)->rxInterruptMask |= mask;
        break;
      default: ASSERT(NULL);
    }
  }
}

static void DMAHwEmulator_InterruptDisable (void * instance,
                                            DMAIRQMask mask,
                                            DMATransferDirection direction)
{
  ASSERT(instance != NULL);
  if (instance != NULL)
  {
    switch (direction)
    {
      case MEMORY_TO_HARDWARE:
        ((DMAHwEmulator *) instance)->txInterruptMask &= ~mask;
        break;
      case HARDWARE_TO_MEMORY:
        ((DMAHwEmulator *) instance)->rxInterruptMask &= ~mask;
        break;
      default: ASSERT(NULL);
    }
  }
}

static void DMAHwEmulator_InterruptClear(void * instance,
                                         DMAIRQMask mask,
                                         DMATransferDirection direction)
{

}

static DMAIRQMask DMAHwEmulator_InterruptGetEnabled (void * instance,
                                                   DMATransferDirection direction)
{
  return DMA_IRQ_IOC;
}

static DMAIRQMask DMAHwEmulator_InterruptGetStatus (void * instance,
                                                  DMATransferDirection direction)
{
  return DMA_IRQ_IOC;
}

void DMAHwEmulator_Reset (void * instance)
{

}

int DMAHwEmulator_ResetIsDone (void * instance)
{
  return 1;
}

static uint32_t  DMAHwEmulator_InterruptSetHandler (void *instance,
                                                    uint32_t interruptId,
                                                    ARM_GIC_InterruptHandler handler,
                                                    void * data)
{
  ASSERT (instance != NULL);

  if (instance != NULL)
  {
    DMAHwEmulator * dma_emulator = (DMAHwEmulator *) instance;
    dma_emulator->interruptVectorId = interruptId;
    dma_emulator->interruptHandler = handler;
    dma_emulator->interruptContextData = data;
  }
  else return XST_FAILURE;

  return XST_SUCCESS;
}

DMAHardware DMAHardware_HwMoverEmulator =
{
  .new =                  DMAHwEmulator_new,
  .delete =               DMAHwEmulator_delete,
  .Initialize =           DMAHwEmulator_Initialize,
  .Move =                 DMAHwEmulator_Move,
  .InterruptEnable =      DMAHwEmulator_InterruptEnable,
  .InterruptDisable =     DMAHwEmulator_InterruptDisable,
  .InterruptClear =       DMAHwEmulator_InterruptClear,
  .InterruptGetEnabled =  DMAHwEmulator_InterruptGetEnabled,
  .InterruptGetStatus =   DMAHwEmulator_InterruptGetStatus,
  .Reset =                DMAHwEmulator_Reset,
  .ResetIsDone =          DMAHwEmulator_ResetIsDone,
  .InterruptSetHandler =  DMAHwEmulator_InterruptSetHandler
};

