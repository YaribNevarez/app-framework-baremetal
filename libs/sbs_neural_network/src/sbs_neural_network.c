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

#include "xscugic.h"

#include "xsbs_spike_50.h"
#include "xsbs_accelerator_10.h"
#include "xsbs_accelerator_1024.h"
#include "xsbs_accelerator_32.h"
#include "xsbs_accelerator_64.h"

#define DEBUG

#ifdef DEBUG

void sbs_assert(const char * file, int line, const char * function, const char * expression)
{
  int BypassFail = 0;
  printf ("Fail: %s, in \"%s\" [%s, %d]\n", expression, function, file, line);

  while (!BypassFail)
    ;

  printf ("Bypass Fail\n");
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
  void *    (*new)(void);
  void      (*delete)(void ** InstancePtr);

  int       (*Initialize) (void *InstancePtr, u16 deviceId);
  void      (*Start)      (void *InstancePtr);
  uint32_t  (*IsDone)     (void *InstancePtr);
  uint32_t  (*IsIdle)     (void *InstancePtr);
  uint32_t  (*IsReady)    (void *InstancePtr);
  void      (*EnableAutoRestart) (void *InstancePtr);
  void      (*DisableAutoRestart) (void *InstancePtr);

  void      (*Set_mode)       (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_mode)       (void *InstancePtr);
  void      (*Set_layerSize)  (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_layerSize)  (void *InstancePtr);
  void      (*Set_kernelSize) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_kernelSize) (void *InstancePtr);
  void      (*Set_vectorSize) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_vectorSize) (void *InstancePtr);
  void      (*Set_epsilon)    (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_epsilon)    (void *InstancePtr);

  void      (*InterruptGlobalEnable)  (void *InstancePtr);
  void      (*InterruptGlobalDisable) (void *InstancePtr);
  void      (*InterruptEnable)        (void *InstancePtr, uint32_t Mask);
  void      (*InterruptDisable)       (void *InstancePtr, uint32_t Mask);
  void      (*InterruptClear)         (void *InstancePtr, uint32_t Mask);
  uint32_t  (*InterruptGetEnabled)    (void *InstancePtr);
  uint32_t  (*InterruptGetStatus)     (void *InstancePtr);
} SbsHardwareDriver;

void Sbs_driver_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}

void * Sbs_spike_50_new (void)
{
  return malloc (sizeof(XSbs_spike_50));
}

SbsHardwareDriver SbsHardwareDriver_spike = {
  .new = Sbs_spike_50_new,
  .delete = Sbs_driver_delete,
  .Initialize = XSbs_spike_50_Initialize,

  .Start = XSbs_spike_50_Start,
  .IsDone = XSbs_spike_50_IsDone,
  .IsIdle = XSbs_spike_50_IsIdle,
  .IsReady = XSbs_spike_50_IsReady,
  .EnableAutoRestart = XSbs_spike_50_EnableAutoRestart,
  .DisableAutoRestart = XSbs_spike_50_DisableAutoRestart,

  .Set_mode = NULL,
  .Get_mode = NULL,
  .Set_layerSize = XSbs_spike_50_Set_layerSize,
  .Get_layerSize = XSbs_spike_50_Get_layerSize,
  .Set_kernelSize = NULL,
  .Get_kernelSize = NULL,
  .Set_vectorSize = XSbs_spike_50_Set_vectorSize,
  .Get_vectorSize = XSbs_spike_50_Get_vectorSize,
  .Set_epsilon = NULL,
  .Get_epsilon = NULL,

  .InterruptGlobalEnable = XSbs_spike_50_InterruptGlobalEnable,
  .InterruptGlobalDisable = XSbs_spike_50_InterruptGlobalDisable,
  .InterruptEnable = XSbs_spike_50_InterruptEnable,
  .InterruptDisable = XSbs_spike_50_InterruptDisable,
  .InterruptClear = XSbs_spike_50_InterruptClear,
  .InterruptGetEnabled = XSbs_spike_50_InterruptGetEnabled,
  .InterruptGetStatus = XSbs_spike_50_InterruptGetStatus,
};


void * Sbs_update_10_new (void)
{
  return malloc (sizeof(XSbs_accelerator_10));
}

SbsHardwareDriver SbsHardwareDriver_update10 = {
    .new = Sbs_update_10_new,
    .delete = Sbs_driver_delete,

    .Initialize = XSbs_accelerator_10_Initialize,
    .Start = XSbs_accelerator_10_Start,
    .IsDone = XSbs_accelerator_10_IsDone,
    .IsIdle = XSbs_accelerator_10_IsIdle,
    .IsReady = XSbs_accelerator_10_IsReady,
    .EnableAutoRestart = XSbs_accelerator_10_EnableAutoRestart,
    .DisableAutoRestart = XSbs_accelerator_10_DisableAutoRestart,

    .Set_mode = XSbs_accelerator_10_Set_mode,
    .Get_mode = XSbs_accelerator_10_Get_mode,
    .Set_layerSize = XSbs_accelerator_10_Set_layerSize,
    .Get_layerSize = XSbs_accelerator_10_Get_layerSize,
    .Set_kernelSize = XSbs_accelerator_10_Set_kernelSize,
    .Get_kernelSize = XSbs_accelerator_10_Get_kernelSize,
    .Set_vectorSize = XSbs_accelerator_10_Set_vectorSize,
    .Get_vectorSize = XSbs_accelerator_10_Get_vectorSize,
    .Set_epsilon = XSbs_accelerator_10_Set_epsilon,
    .Get_epsilon = XSbs_accelerator_10_Get_epsilon,

    .InterruptGlobalEnable = XSbs_accelerator_10_InterruptGlobalEnable,
    .InterruptGlobalDisable = XSbs_accelerator_10_InterruptGlobalDisable,
    .InterruptEnable = XSbs_accelerator_10_InterruptEnable,
    .InterruptDisable = XSbs_accelerator_10_InterruptDisable,
    .InterruptClear = XSbs_accelerator_10_InterruptClear,
    .InterruptGetEnabled = XSbs_accelerator_10_InterruptGetEnabled,
    .InterruptGetStatus = XSbs_accelerator_10_InterruptGetStatus
  };


void * Sbs_update_1024_new (void)
{
  return malloc (sizeof(XSbs_accelerator_1024));
}

SbsHardwareDriver SbsHardwareDriver_update1024 = {
    .new = Sbs_update_1024_new,
    .delete = Sbs_driver_delete,

    .Initialize = XSbs_accelerator_1024_Initialize,
    .Start = XSbs_accelerator_1024_Start,
    .IsDone = XSbs_accelerator_1024_IsDone,
    .IsIdle = XSbs_accelerator_1024_IsIdle,
    .IsReady = XSbs_accelerator_1024_IsReady,
    .EnableAutoRestart = XSbs_accelerator_1024_EnableAutoRestart,
    .DisableAutoRestart = XSbs_accelerator_1024_DisableAutoRestart,

    .Set_mode = XSbs_accelerator_1024_Set_mode,
    .Get_mode = XSbs_accelerator_1024_Get_mode,
    .Set_layerSize = XSbs_accelerator_1024_Set_layerSize,
    .Get_layerSize = XSbs_accelerator_1024_Get_layerSize,
    .Set_kernelSize = XSbs_accelerator_1024_Set_kernelSize,
    .Get_kernelSize = XSbs_accelerator_1024_Get_kernelSize,
    .Set_vectorSize = XSbs_accelerator_1024_Set_vectorSize,
    .Get_vectorSize = XSbs_accelerator_1024_Get_vectorSize,
    .Set_epsilon = XSbs_accelerator_1024_Set_epsilon,
    .Get_epsilon = XSbs_accelerator_1024_Get_epsilon,

    .InterruptGlobalEnable = XSbs_accelerator_1024_InterruptGlobalEnable,
    .InterruptGlobalDisable = XSbs_accelerator_1024_InterruptGlobalDisable,
    .InterruptEnable = XSbs_accelerator_1024_InterruptEnable,
    .InterruptDisable = XSbs_accelerator_1024_InterruptDisable,
    .InterruptClear = XSbs_accelerator_1024_InterruptClear,
    .InterruptGetEnabled = XSbs_accelerator_1024_InterruptGetEnabled,
    .InterruptGetStatus = XSbs_accelerator_1024_InterruptGetStatus
  };


void * Sbs_update_32_new (void)
{
  return malloc (sizeof(XSbs_accelerator_32));
}

SbsHardwareDriver SbsHardwareDriver_update32 = {
    .new = Sbs_update_32_new,
    .delete = Sbs_driver_delete,

    .Initialize = XSbs_accelerator_32_Initialize,
    .Start = XSbs_accelerator_32_Start,
    .IsDone = XSbs_accelerator_32_IsDone,
    .IsIdle = XSbs_accelerator_32_IsIdle,
    .IsReady = XSbs_accelerator_32_IsReady,
    .EnableAutoRestart = XSbs_accelerator_32_EnableAutoRestart,
    .DisableAutoRestart = XSbs_accelerator_32_DisableAutoRestart,

    .Set_mode = XSbs_accelerator_32_Set_mode,
    .Get_mode = XSbs_accelerator_32_Get_mode,
    .Set_layerSize = XSbs_accelerator_32_Set_layerSize,
    .Get_layerSize = XSbs_accelerator_32_Get_layerSize,
    .Set_kernelSize = XSbs_accelerator_32_Set_kernelSize,
    .Get_kernelSize = XSbs_accelerator_32_Get_kernelSize,
    .Set_vectorSize = XSbs_accelerator_32_Set_vectorSize,
    .Get_vectorSize = XSbs_accelerator_32_Get_vectorSize,
    .Set_epsilon = XSbs_accelerator_32_Set_epsilon,
    .Get_epsilon = XSbs_accelerator_32_Get_epsilon,

    .InterruptGlobalEnable = XSbs_accelerator_32_InterruptGlobalEnable,
    .InterruptGlobalDisable = XSbs_accelerator_32_InterruptGlobalDisable,
    .InterruptEnable = XSbs_accelerator_32_InterruptEnable,
    .InterruptDisable = XSbs_accelerator_32_InterruptDisable,
    .InterruptClear = XSbs_accelerator_32_InterruptClear,
    .InterruptGetEnabled = XSbs_accelerator_32_InterruptGetEnabled,
    .InterruptGetStatus = XSbs_accelerator_32_InterruptGetStatus
  };


void * Sbs_update_64_new (void)
{
  return malloc (sizeof(XSbs_accelerator_64));
}

SbsHardwareDriver SbsHardwareDriver_update64 = {
    .new = Sbs_update_64_new,
    .delete = Sbs_driver_delete,

    .Initialize = XSbs_accelerator_64_Initialize,
    .Start = XSbs_accelerator_64_Start,
    .IsDone = XSbs_accelerator_64_IsDone,
    .IsIdle = XSbs_accelerator_64_IsIdle,
    .IsReady = XSbs_accelerator_64_IsReady,
    .EnableAutoRestart = XSbs_accelerator_64_EnableAutoRestart,
    .DisableAutoRestart = XSbs_accelerator_64_DisableAutoRestart,

    .Set_mode = XSbs_accelerator_64_Set_mode,
    .Get_mode = XSbs_accelerator_64_Get_mode,
    .Set_layerSize = XSbs_accelerator_64_Set_layerSize,
    .Get_layerSize = XSbs_accelerator_64_Get_layerSize,
    .Set_kernelSize = XSbs_accelerator_64_Set_kernelSize,
    .Get_kernelSize = XSbs_accelerator_64_Get_kernelSize,
    .Set_vectorSize = XSbs_accelerator_64_Set_vectorSize,
    .Get_vectorSize = XSbs_accelerator_64_Get_vectorSize,
    .Set_epsilon = XSbs_accelerator_64_Set_epsilon,
    .Get_epsilon = XSbs_accelerator_64_Get_epsilon,

    .InterruptGlobalEnable = XSbs_accelerator_64_InterruptGlobalEnable,
    .InterruptGlobalDisable = XSbs_accelerator_64_InterruptGlobalDisable,
    .InterruptEnable = XSbs_accelerator_64_InterruptEnable,
    .InterruptDisable = XSbs_accelerator_64_InterruptDisable,
    .InterruptClear = XSbs_accelerator_64_InterruptClear,
    .InterruptGetEnabled = XSbs_accelerator_64_InterruptGetEnabled,
    .InterruptGetStatus = XSbs_accelerator_64_InterruptGetStatus
  };

//void * Sbs_update_64_p_new (void)
//{
//  return malloc (sizeof(XSbs_accelerator_64_p));
//}
//
//SbsHardwareDriver SbsHardwareDriver_update64p = {
//    .new = Sbs_update_64_p_new,
//    .delete = Sbs_driver_delete,
//
//    .Initialize = XSbs_accelerator_64_p_Initialize,
//    .Start = XSbs_accelerator_64_p_Start,
//    .IsDone = XSbs_accelerator_64_p_IsDone,
//    .IsIdle = XSbs_accelerator_64_p_IsIdle,
//    .IsReady = XSbs_accelerator_64_p_IsReady,
//    .EnableAutoRestart = XSbs_accelerator_64_p_EnableAutoRestart,
//    .DisableAutoRestart = XSbs_accelerator_64_p_DisableAutoRestart,
//
//    .Set_layerSize = XSbs_accelerator_64_p_Set_layerSize,
//    .Get_layerSize = XSbs_accelerator_64_p_Get_layerSize,
//    .Set_kernelSize = XSbs_accelerator_64_p_Set_kernelSize,
//    .Get_kernelSize = XSbs_accelerator_64_p_Get_kernelSize,
//    .Set_vectorSize = XSbs_accelerator_64_p_Set_vectorSize,
//    .Get_vectorSize = XSbs_accelerator_64_p_Get_vectorSize,
//    .Set_epsilon = XSbs_accelerator_64_p_Set_epsilon,
//    .Get_epsilon = XSbs_accelerator_64_p_Get_epsilon,
//
//    .InterruptGlobalEnable = XSbs_accelerator_64_p_InterruptGlobalEnable,
//    .InterruptGlobalDisable = XSbs_accelerator_64_p_InterruptGlobalDisable,
//    .InterruptEnable = XSbs_accelerator_64_p_InterruptEnable,
//    .InterruptDisable = XSbs_accelerator_64_p_InterruptDisable,
//    .InterruptClear = XSbs_accelerator_64_p_InterruptClear,
//    .InterruptGetEnabled = XSbs_accelerator_64_p_InterruptGetEnabled,
//    .InterruptGetStatus = XSbs_accelerator_64_p_InterruptGetStatus
//  };

typedef struct
{
  SbsHardwareDriver * hwDriver;
  SbsLayerType        layerType;
  uint32_t            vectorSize;
  uint32_t            hwDeviceID;
  uint32_t            dmaDeviceID;
  uint32_t            hwIntVecID;
  uint32_t            dmaTxIntVecID;
  uint32_t            dmaRxIntVecID;
  MemoryBlock         ddrMem;
} SbSHardwareConfig;

typedef enum
{
  SPIKE_MODE  = 0,
  UPDATE_MODE = 1,
  ACCELERATOR_MODES
} AcceleratorMode;

typedef struct
{
  uint32_t      layerSize;
  uint32_t      kernelSize;
  uint32_t      vectorSize;
  float         epsilon;

  size_t        vectorBufferSize;

  size_t        txBufferSize[ACCELERATOR_MODES];
  uint32_t *    rxBuffer[ACCELERATOR_MODES];
  size_t        rxBufferSize[ACCELERATOR_MODES];
} SbsAcceleratorProfie;

typedef struct
{
  SbSHardwareConfig *     hardwareConfig;
  void *                  updateHardware;
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
typedef float     Random32;
typedef uint32_t  SpikeID;

typedef enum
{
  M32BIT_TYPE_BEGIN = 0,
  M32BIT_24_24_ID,
  M32BIT_24_24_50_ID,
  M32BIT_24_24_32_ID,
  M32BIT_12_24_32_ID,
  M32BIT_1_1_50_32_ID,
  M32BIT_6_12_32_ID,
  M32BIT_12_12_32_ID,
  M32BIT_12_12_ID,
  M32BIT_2_2_32_32_ID,
  M32BIT_8_8_64_ID,
  M32BIT_8_8_ID,
  M32BIT_5_5_32_64_ID,
  M32BIT_2_4_64_ID,
  M32BIT_4_4_64_ID,
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
typedef uint32_t M32Bit_24_24_32[24][24][32];
typedef uint32_t M32Bit_1_1_50_32[1][1][50][32];
typedef uint32_t M32Bit_6_12_32[6][12][32];
typedef uint32_t M32Bit_12_12_32[6][12][32];
typedef uint32_t M32Bit_12_12[12][12];
typedef uint32_t M32Bit_2_2_32_32[2][2][32][32];
typedef uint32_t M32Bit_8_8_64[8][8][64];
typedef uint32_t M32Bit_8_8[8][8];
typedef uint32_t M32Bit_5_5_32_64[5][5][32][64];
typedef uint32_t M32Bit_2_4_64[2][4][64];
typedef uint32_t M32Bit_4_4_64[4][4][64];
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
        .type_id = M32BIT_24_24_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {24, 24, 32, 0}
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
        .type_id = M32BIT_12_12_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {12, 12, 32, 0}
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
        .type_id = M32BIT_4_4_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {4, 4, 64, 0}
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
  Multivector * spike_matrix;
  Multivector * weight_matrix;
} SbsLayerPartition;

typedef struct
{
  SbsLayer              vtbl;
  SbsLayerType          layer_type;
  SbsLayerPartition **  partition_array;
  uint8_t               num_partitions;
  uint16_t              rows;
  uint16_t              columns;
  uint16_t              vector_size;
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

SbSHardwareConfig SbSHardwareConfig_list[] =
{
  { .hwDriver      = &SbsHardwareDriver_spike,
    .layerType     = INPUT_LAYER_50N,
    .vectorSize    = 50,
    .hwDeviceID    = XPAR_XSBS_SPIKE_50_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_SPIKE_50_0_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_0_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x24000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x27FFFFFF,
      .blockIndex  = 0
    }
  },
  { .hwDriver      = &SbsHardwareDriver_update32,
    .layerType     = CONVOLUTION_LAYER_32N,
    .vectorSize    = 32,
    .hwDeviceID    = XPAR_XSBS_ACCELERATOR_32_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_1_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_32_0_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_1_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x28000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x2BFFFFFF,
      .blockIndex  = 0
    }
  },
  { .hwDriver      = &SbsHardwareDriver_update32,
    .layerType     = POOLING_LAYER_32N,
    .vectorSize    = 32,
    .hwDeviceID    = XPAR_XSBS_ACCELERATOR_32_1_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_2_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_32_1_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_2_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x2C000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x2FFFFFFF,
      .blockIndex  = 0
    }
  },
  { .hwDriver      = &SbsHardwareDriver_update64,
    .layerType     = CONVOLUTION_LAYER_64N,
    .vectorSize    = 64,
    .hwDeviceID    = XPAR_SBS_ACCELERATOR_64_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_3_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_64_0_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_3_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x30000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x33FFFFFF,
      .blockIndex  = 0
    }
  },
  { .hwDriver      = &SbsHardwareDriver_update64,
    .layerType     = POOLING_LAYER_64N,
    .vectorSize    = 64,
    .hwDeviceID    = XPAR_SBS_ACCELERATOR_64_1_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_4_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_64_1_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_4_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x34000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x37FFFFFF,
      .blockIndex  = 0
    }
  },
  { .hwDriver      = &SbsHardwareDriver_update1024,
    .layerType     = FULLY_CONNECTED_LAYER_1024N,
    .vectorSize    = 1024,
    .hwDeviceID    = XPAR_SBS_ACCELERATOR_1024_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_5_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_1024_0_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_5_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x38000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3BFFFFFF,
      .blockIndex  = 0
    }
  },
  { .hwDriver      = &SbsHardwareDriver_update10,
    .layerType     = OUTPUT_LAYER_10N,
    .vectorSize    = 10,
    .hwDeviceID    = XPAR_SBS_ACCELERATOR_10_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_AXIDMA_6_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_10_0_VEC_ID,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = XPAR_FABRIC_AXIDMA_6_VEC_ID,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3C000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x3FFFFFFF,
      .blockIndex  = 0
    }
  }
};

#define NUM_ACCELERATOR_INSTANCES  (sizeof(SbSHardwareConfig_list) / sizeof(SbSHardwareConfig))

static SbSUpdateAccelerator *  SbSUpdateAccelerator_list[NUM_ACCELERATOR_INSTANCES] = {NULL};

static SbSUpdateAccelerator * SbSUpdateAccelerator_getFromList (SbsLayerType layerType)
{
  SbSUpdateAccelerator * accelerator = NULL;
  int i;
  ASSERT (SbSUpdateAccelerator_list != NULL);
  for (i = 0; accelerator == NULL && i < NUM_ACCELERATOR_INSTANCES; i++)
  {
    ASSERT(SbSUpdateAccelerator_list[i] != NULL);
    ASSERT(SbSUpdateAccelerator_list[i]->hardwareConfig != NULL);
    if (SbSUpdateAccelerator_list[i] != NULL
        && SbSUpdateAccelerator_list[i]->hardwareConfig->layerType == layerType)
      accelerator = SbSUpdateAccelerator_list[i];
  }
  return accelerator;
}




static XScuGic                 ScuGic = {0};

#define ACCELERATOR_DMA_RESET_TIMEOUT 10000

static void Accelerator_txInterruptHandler(void * data)
{
  XAxiDma *AxiDmaInst = &((SbSUpdateAccelerator *) data)->dmaHardware;
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
  XAxiDma *AxiDmaInst = &((SbSUpdateAccelerator *) data)->dmaHardware;
  uint32_t IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DEVICE_TO_DMA);

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
    ((SbSUpdateAccelerator *) data)->txDone = 1; // TODO: Implement tx interrupt
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

static int Accelerator_initialize(SbSUpdateAccelerator * accelerator,
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
  ASSERT (NULL != IntcConfig);
  if (NULL == IntcConfig)
  {
    return XST_FAILURE;
  }

  status = XScuGic_CfgInitialize (&ScuGic, IntcConfig,
                                  IntcConfig->CpuBaseAddress);
  ASSERT (status == XST_SUCCESS);
  if (status != XST_SUCCESS)
  {
    return XST_FAILURE;
  }

  if (hardware_config->dmaTxIntVecID)
  {
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

  if (hardware_config->dmaRxIntVecID)
  {
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

  if (hardware_config->hwIntVecID)
  {
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

  /**************************** initialize ARM Core exception handlers *******/
  Xil_ExceptionInit ();
  Xil_ExceptionRegisterHandler (XIL_EXCEPTION_ID_INT,
                                (Xil_ExceptionHandler) XScuGic_InterruptHandler,
                                (void *) &ScuGic);

  Xil_ExceptionEnable();

  /***************************************************************************/
  /**************************** Accelerator initialization *******************/

  accelerator->updateHardware = hardware_config->hwDriver->new();

  ASSERT (accelerator->updateHardware != NULL);

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
    if (accelerator->hardwareConfig->dmaTxIntVecID)
      XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->dmaTxIntVecID);

    if (accelerator->hardwareConfig->dmaRxIntVecID)
      XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->dmaRxIntVecID);

    if (accelerator->hardwareConfig->hwIntVecID)
      XScuGic_Disconnect (&ScuGic, accelerator->hardwareConfig->hwIntVecID);
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
    (*accelerator)->hardwareConfig->hwDriver->delete(&(*accelerator)->updateHardware);
    free (*accelerator);
    *accelerator = NULL;
  }
}

static void Accelerator_setup(SbSUpdateAccelerator * accelerator,
                              SbsAcceleratorProfie * profile,
                              AcceleratorMode mode)
{
  ASSERT (accelerator != NULL);
  ASSERT (profile != NULL);

  ASSERT (accelerator->hardwareConfig != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver != NULL);

  //if (accelerator->profile != profile)
  {
    accelerator->profile = profile;

    if (accelerator->hardwareConfig->hwDriver->Set_layerSize)
      accelerator->hardwareConfig->hwDriver->Set_layerSize (
          accelerator->updateHardware, profile->layerSize);

    if (accelerator->hardwareConfig->hwDriver->Set_kernelSize)
      accelerator->hardwareConfig->hwDriver->Set_kernelSize (
          accelerator->updateHardware, profile->kernelSize);

    if (accelerator->hardwareConfig->hwDriver->Set_vectorSize)
      accelerator->hardwareConfig->hwDriver->Set_vectorSize (
          accelerator->updateHardware, profile->vectorSize);

    if (accelerator->hardwareConfig->hwDriver->Set_epsilon)
      accelerator->hardwareConfig->hwDriver->Set_epsilon (
          accelerator->updateHardware, *(uint32_t*) &profile->epsilon);
  }

  ASSERT(accelerator->txBuffer != NULL);

  /************************** Rx Setup **************************/

  accelerator->rxBuffer = profile->rxBuffer[mode];
  accelerator->rxBufferSize = profile->rxBufferSize[mode];

  /************************** Tx Setup **************************/

  accelerator->txBufferSize = profile->txBufferSize[mode];

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

  ((Random32 *)accelerator->txBuffer)[accelerator->txBufferIndex ++] =
      ((NeuronState) genrand ()) * (1.0/((NeuronState) 0xFFFFFFFF));;

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
  ASSERT (0 < accelerator->profile->layerSize);

#ifdef DEBUG
  ASSERT (accelerator->profile->layerSize == accelerator->txStateCounter);
  ASSERT (accelerator->profile->kernelSize * accelerator->profile->layerSize == accelerator->txWeightCounter);
#endif

  Xil_DCacheFlushRange ((UINTPTR) accelerator->txBuffer, accelerator->txBufferSize);

  while (accelerator->txDone == 0) tx_wait[layer_wait][accelerator->hardwareConfig->dmaDeviceID] ++;
  status = XAxiDma_SimpleTransfer (&accelerator->dmaHardware,
                                   (UINTPTR) accelerator->txBuffer,
                                   accelerator->txBufferSize,
                                   XAXIDMA_DMA_TO_DEVICE);
  ASSERT(status == XST_SUCCESS);
  accelerator->txDone = 0;

  while (accelerator->rxDone == 0) rx_wait[layer_wait][accelerator->hardwareConfig->dmaDeviceID] ++;
  status = XAxiDma_SimpleTransfer (&accelerator->dmaHardware,
                                   (UINTPTR) accelerator->rxBuffer,
                                   accelerator->rxBufferSize,
                                   XAXIDMA_DEVICE_TO_DMA);
  ASSERT(status == XST_SUCCESS);
  accelerator->rxDone = 0;

  while (accelerator->acceleratorReady == 0) accelerator_wait[layer_wait][accelerator->hardwareConfig->dmaDeviceID] ++;
  accelerator->hardwareConfig->hwDriver->Start (accelerator->updateHardware);
  accelerator->acceleratorReady = 0;
}

/*****************************************************************************/

Result SbsHardware_initialize (void)
{
  int i;
  Result rc;

  rc = (SbSUpdateAccelerator_list != NULL) ? OK : ERROR;

  for (i = 0; (rc == OK) && (i < NUM_ACCELERATOR_INSTANCES); i++)
  {
    SbSUpdateAccelerator_list[i] = Accelerator_new (&SbSHardwareConfig_list[i]);

    ASSERT (SbSUpdateAccelerator_list[i] != NULL);

    rc = SbSUpdateAccelerator_list[i] != NULL ? OK : ERROR;
  }

  return rc;
}

void SbsHardware_shutdown (void)
{
  int i;
  ASSERT (SbSUpdateAccelerator_list != NULL);

  if (SbSUpdateAccelerator_list != NULL)
  {
    for (i = 0; i < NUM_ACCELERATOR_INSTANCES; i++)
    {
      Accelerator_delete ((&SbSUpdateAccelerator_list[i]));
    }
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
    case M32BIT_24_24_32_ID:
      return &(*(M32Bit_24_24_32*) multivector->data)[row][column];
    case M32BIT_1_1_50_32_ID:
      return &(*(M32Bit_1_1_50_32*) multivector->data)[row][column];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column];
    case M32BIT_12_12_32_ID:
      return &(*(M32Bit_12_12_32*) multivector->data)[row][column];
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
    case M32BIT_4_4_64_ID:
      return &(*(M32Bit_4_4_64*) multivector->data)[row][column];
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

static size_t Multivector_dataSize(Multivector * multivector)
{
  size_t data_size = 0;
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    int i;
    data_size = multivector->data_type_size;

    for (i = 0; i < multivector->dimensionality; i++)
      data_size *= multivector->dimension_size[i];
  }
  return data_size;
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
                                     SbsLayerType layerType,
                                     Multivector * state_matrix,
                                     Multivector * spike_matrix,
                                     uint32_t kernel_size,
                                     float epsilon)
{
  ASSERT (profile != NULL);
  ASSERT (state_matrix != NULL);
  ASSERT (state_matrix->dimensionality == 3);
  ASSERT (state_matrix->data != NULL);

  if ((profile != NULL)
      && (state_matrix != NULL)
      && (state_matrix->dimensionality == 3)
      && (state_matrix->data != NULL))
  {
    profile->layerSize =
        state_matrix->dimension_size[0] * state_matrix->dimension_size[1];
    profile->vectorSize = state_matrix->dimension_size[2];

    if (layerType == INPUT_LAYER_50N)
    {
      profile->kernelSize = 0;
      profile->epsilon = 0;
    }
    else
    {
      ASSERT (0 < kernel_size);
      ASSERT (0.0 < epsilon);

      profile->kernelSize = kernel_size * kernel_size;
      profile->epsilon = epsilon;
    }

    profile->rxBuffer[SPIKE_MODE] = spike_matrix->data;
    profile->rxBufferSize[SPIKE_MODE] = Multivector_dataSize(spike_matrix);
    profile->txBufferSize[SPIKE_MODE] = profile->layerSize
        * (sizeof(Random32) + profile->vectorSize * state_matrix->data_type_size);

    profile->rxBuffer[UPDATE_MODE] = state_matrix->data;
    profile->rxBufferSize[UPDATE_MODE] = Multivector_dataSize(state_matrix);
    profile->txBufferSize[UPDATE_MODE] = profile->layerSize
    * (sizeof(Random32) + (1 + profile->kernelSize) * profile->vectorSize * state_matrix->data_type_size);

    profile->vectorBufferSize = profile->vectorSize * state_matrix->data_type_size;
  }
}

/*****************************************************************************/
static SbsLayerPartition * SbsLayerPartition_new (SbSUpdateAccelerator * accelerator,
                                                  uint16_t x_pos,
                                                  uint16_t y_pos,
                                                  uint16_t rows,
                                                  uint16_t columns,
                                                  uint16_t vector_size)
{
  SbsLayerPartition * partition = NULL;

  partition = (SbsLayerPartition *) malloc (sizeof(SbsLayerPartition));

  ASSERT (partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;
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
                                    columns, vector_size);

    ASSERT(state_matrix != NULL);
    ASSERT(state_matrix->dimensionality == 3);
    ASSERT(state_matrix->data != NULL);
    ASSERT(state_matrix->dimension_size[0] == rows);
    ASSERT(state_matrix->dimension_size[1] == columns);
    ASSERT(state_matrix->dimension_size[2] == vector_size);

    partition->state_matrix = state_matrix;

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new (memory_def, sizeof(SpikeID), 2, rows, columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    partition->spike_matrix = spike_matrix;

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
    Multivector_delete (&((*partition)->spike_matrix));
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
                                          SbsLayerType layerType,
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

    if (layerType != INPUT_LAYER_50N)
      for (row = 0; row < rows; row++)
      {
        current_row_index = row * columns * neurons;
        for (column = 0; column < columns; column++)
        {
          SbsLayerPartition_initializeIP (
              &state_matrix_data[current_row_index + column * neurons],
              neurons);
        }
      }

    SbsAcceleratorProfie_initialize(&partition->profile,
                                    layerType,
                                    state_matrix,
                                    partition->spike_matrix,
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
                                   uint16_t kernel_size,
                                   uint16_t kernel_stride,
                                   WeightShift weight_shift)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);

  if (layer != NULL)
  {
    int i;
    uint16_t vector_size = 0;
    SbSUpdateAccelerator * accelerator_group[NUM_ACCELERATOR_INSTANCES] = {0};

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    layer->layer_type = layer_type;
    layer->num_partitions = 1;

    accelerator_group[0] = SbSUpdateAccelerator_getFromList(layer_type);

    ASSERT (accelerator_group[0] != NULL);

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
        vector_size = accelerator->hardwareConfig->vectorSize;

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
                                                           vector_size);

        pos_y += rows_current_partition;
      }
    }

    /* Assign parameters */
    layer->rows    = rows;
    layer->columns = columns;
    layer->vector_size   = vector_size;
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
                                    layer->layer_type,
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

  }
}

static void SbsBaseLayer_loadInput (SbsBaseLayer * layer, char * file_name,
                                    uint8_t * input_label)
{
  ASSERT(layer != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  ASSERT(layer->layer_type == INPUT_LAYER_50N);
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
  ASSERT(layer->layer_type == OUTPUT_LAYER_10N);
  ASSERT(layer->partition_array[layer->num_partitions - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != NULL);

  if ((layer != NULL) && (0 < layer->num_partitions)
      && (layer->layer_type == OUTPUT_LAYER_10N)
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

//static void SbsBaseLayer_generateSpikes (SbsBaseLayer * layer)
//{
//  ASSERT(layer != NULL);
//  ASSERT(layer->partition_array != NULL);
//  ASSERT(0 < layer->num_partitions);
//  ASSERT(layer->spike_matrix != NULL);
//
//  if ((layer != NULL)
//      && (layer->partition_array != NULL)
//      && (0 < layer->num_partitions)
//      && (layer->spike_matrix != NULL))
//  {
//    int i;
//    uint16_t columns = layer->columns;
//    uint16_t neurons = layer->neurons;
//
//    uint16_t partition_row = 0;
//    SbsLayerPartition *  partition = NULL;
//    Multivector * partition_state_matrix = NULL;
//    Multivector* layer_spike_matrix = layer->spike_matrix;
//
//    SpikeID * spike;
//    NeuronState * state_vector;
//
//    uint16_t row;
//    uint16_t column;
//
//    for (i = 0; i < layer->num_partitions; i ++)
//    {
//      ASSERT(layer->partition_array[i] != NULL);
//      ASSERT(layer->partition_array[i]->state_matrix != NULL);
//      partition = layer->partition_array[i];
//      partition_state_matrix = partition->state_matrix;
//
//      for (row = partition->y_pos, partition_row = 0;
//          partition_row < partition_state_matrix->dimension_size[0];
//          partition_row++, row ++)
//      {
//        for (column = 0; column < columns; column++)
//        {
//          spike = Multivector_2DAccess (layer_spike_matrix, row, column);
//          state_vector = Multivector_2DAccess (partition_state_matrix,
//                                               partition_row,
//                                               column);
//          *spike = SbsStateVector_generateSpike (state_vector, neurons);
//        }
//      }
//    }
//  }
//}

static void SbsBaseLayer_generateSpikes (SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    int i;
    uint16_t columns = layer->columns;

    uint16_t partition_row = 0;
    SbsLayerPartition *  partition = NULL;
    Multivector * partition_state_matrix = NULL;
    NeuronState * state_vector;

    uint16_t row;
    uint16_t column;

    for (i = 0; i < layer->num_partitions; i ++)
    {
      ASSERT(layer->partition_array[i] != NULL);
      ASSERT(layer->partition_array[i]->state_matrix != NULL);
      ASSERT(layer->partition_array[i]->spike_matrix != NULL);
      partition = layer->partition_array[i];
      partition_state_matrix = partition->state_matrix;

      Accelerator_setup(partition->accelerator, &partition->profile, SPIKE_MODE);

      for (row = partition->y_pos, partition_row = 0;
          partition_row < partition_state_matrix->dimension_size[0];
          partition_row++, row ++)
      {
        for (column = 0; column < columns; column++)
        {
          state_vector = Multivector_2DAccess (partition_state_matrix,
                                               partition_row,
                                               column);
//          *((SpikeID*)Multivector_2DAccess (layer->partition_array[i]->spike_matrix,
//                                                         partition_row,
//                                                         column)) =
//          SbsStateVector_generateSpike(state_vector, layer->vector_size);
          Accelerator_giveStateVector (partition->accelerator, state_vector);
        }
      }

      Accelerator_start (partition->accelerator);
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
    Multivector * spike_layer_spike_matrix = spike_layer->partition_array[0]->spike_matrix;


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

    WeightShift layer_weight_shift = layer->weight_shift;

    kernel_row_pos = 0, layer_row = 0;
    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      update_partition_weight_matrix = update_partition->weight_matrix;
      update_partition_accelerator = update_partition->accelerator;
      update_partition_state_matrix = update_partition->state_matrix;

      Accelerator_setup (update_partition_accelerator,
                         &update_partition->profile,
                         UPDATE_MODE);

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
          while (!spike_layer->partition_array[0]->accelerator->rxDone);
          state_vector = Multivector_2DAccess(update_partition_state_matrix, update_partition_row, layer_column);

          Accelerator_giveStateVector (update_partition_accelerator, state_vector);

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              spikeID = *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

              ASSERT(layer->vector_size == update_partition->weight_matrix->dimension_size[3]);

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
    for (i = 0; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
      SbsBaseLayer_cacheFlush(network->layer_array[i]);
    }

    Timer_start(timer);
    /************************ Begins Update cycle **************************/
    while (cycles--)
    {
//      for (i = 0; i <= network->size - 1; i ++)
//      {
//        if (i == 0)
//          SbsBaseLayer_generateSpikes (network->layer_array[i]);
//
//        layer_wait = i;
//        if (0 < i) SbsBaseLayer_update (network->layer_array[i],
//                                        network->layer_array[i - 1]);
//      }

      for (i = 0; i <= network->size - 1; i ++)
          SbsBaseLayer_generateSpikes (network->layer_array[i]);

      for (i = 1; i <= network->size - 1; i++)
        SbsBaseLayer_update (network->layer_array[i],
                             network->layer_array[i - 1]);

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

static SbsLayer * SbsInputLayer_new(SbsLayerType layer_type, uint16_t rows, uint16_t columns)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type, rows, columns, 0,
                                        0, ROW_SHIFT);
}

static SbsLayer * SbsConvolutionLayer_new(SbsLayerType layer_type,
                                          uint16_t rows,
                                          uint16_t columns,
                                          uint16_t kernel_size,
                                          WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type, rows, columns,
                                        kernel_size, 1, weight_shift);
}

static SbsLayer * SbsPoolingLayer_new(SbsLayerType layer_type,
                                      uint16_t rows,
                                      uint16_t columns,
                                      uint16_t kernel_size,
                                      WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type, rows, columns,
                                        kernel_size, kernel_size, weight_shift);
}

static SbsLayer * SbsFullyConnectedLayer_new(SbsLayerType layer_type,
                                             uint16_t kernel_size,
                                             WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type, 1, 1,
                                        kernel_size, 1, weight_shift);
}

static SbsLayer * SbsOutputLayer_new(SbsLayerType layer_type,
                                     WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type, 1, 1, 1, 1,
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
