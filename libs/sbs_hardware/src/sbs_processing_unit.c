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

int SbSUpdateAccelerator_getGroupFromList (SbsLayerType layerType,
                                           SbSUpdateAccelerator ** sub_list,
                                           int sub_list_size)
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




#define ACCELERATOR_DMA_RESET_TIMEOUT 10000

static void Accelerator_txInterruptHandler (void * data)
{
  DMAHardware * driver  = ((SbSUpdateAccelerator *) data)->hardwareConfig->dmaDriver;
  void *        dma     = ((SbSUpdateAccelerator *) data)->dmaHardware;
  DMAIRQMask irq_status = driver->InterruptGetStatus (dma, MEMORY_TO_HARDWARE);

  driver->InterruptClear (dma, irq_status, MEMORY_TO_HARDWARE);

  if (!(irq_status & DMA_IRQ_ALL)) return;

  if (irq_status & DMA_IRQ_DELAY) return;

  if (irq_status & DMA_IRQ_ERROR)
  {
    int TimeOut;

    ((SbSUpdateAccelerator *) data)->errorFlags |= 0x01;

    driver->Reset (dma);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (driver->ResetIsDone (dma)) break;

    ASSERT(0);
    return;
  }

  if (irq_status & DMA_IRQ_IOC) ((SbSUpdateAccelerator *) data)->txDone = 1;
}

static void Accelerator_rxInterruptHandler (void * data)
{
  SbSUpdateAccelerator *  accelerator = (SbSUpdateAccelerator *) data;
  DMAHardware *           driver      = accelerator->hardwareConfig->dmaDriver;
  void *                  dma         = accelerator->dmaHardware;
  DMAIRQMask              irq_status  = driver->InterruptGetStatus (dma, HARDWARE_TO_MEMORY);

  driver->InterruptClear (dma, irq_status, HARDWARE_TO_MEMORY);

  if (!(irq_status & DMA_IRQ_ALL)) return;

  if (irq_status & DMA_IRQ_DELAY) return;

  if (irq_status & DMA_IRQ_ERROR)
  {
    int TimeOut;

    accelerator->errorFlags |= 0x01;

    driver->Reset (dma);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (driver->ResetIsDone (dma)) break;

    ASSERT(0);
    return;
  }

  if (irq_status & DMA_IRQ_IOC)
  {
    Xil_DCacheInvalidateRange ((INTPTR) accelerator->rxBuffer,
                               accelerator->rxBufferSize);

    accelerator->txDone = 1;
    accelerator->rxDone = 1;

    if (accelerator->memory_cmd.cmdID == MEM_CMD_MOVE)
      memcpy (accelerator->memory_cmd.dest,
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

#ifdef DEBUG
  if (accelerator->hardwareConfig->hwDriver->Get_debug)
  {
    //int debug = accelerator->hardwareConfig->hwDriver->Get_debug(accelerator->updateHardware);
    //printf ("\nHW interrupt = 0x%X\n", debug);
  }
#endif

  status = accelerator->hardwareConfig->hwDriver->InterruptGetStatus (accelerator->updateHardware);
  accelerator->hardwareConfig->hwDriver->InterruptClear (accelerator->updateHardware, status);
  accelerator->acceleratorReady = status & 1;
}

int Accelerator_initialize(SbSUpdateAccelerator * accelerator,
                                  SbSHardwareConfig * hardware_config)
{
  int                 status;

  ASSERT (accelerator != NULL);
  ASSERT (hardware_config != NULL);

  if (accelerator == NULL || hardware_config == NULL)
    return XST_FAILURE;

  memset (accelerator, 0x00, sizeof(SbSUpdateAccelerator));

  accelerator->hardwareConfig = hardware_config;

  /******************************* DMA initialization ************************/

  accelerator->dmaHardware = hardware_config->dmaDriver->new ();

  ASSERT(accelerator->dmaHardware != NULL);
  if (accelerator->dmaHardware == NULL) return XST_FAILURE;


  status = hardware_config->dmaDriver->Initialize (accelerator->dmaHardware,
                                                   hardware_config->dmaDeviceID);

  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS) return status;

  if (hardware_config->dmaTxIntVecID)
  {
    hardware_config->dmaDriver->InterruptEnable (accelerator->dmaHardware,
                                                 DMA_IRQ_ALL,
                                                 MEMORY_TO_HARDWARE);

    status = hardware_config->dmaDriver->InterruptSetHandler (accelerator->dmaHardware,
                                                              hardware_config->dmaTxIntVecID,
                                                              Accelerator_txInterruptHandler,
                                                              accelerator);
    ASSERT(status != XST_SUCCESS);
    if (status != XST_SUCCESS) return status;
  }

  if (hardware_config->dmaRxIntVecID)
  {
    hardware_config->dmaDriver->InterruptEnable (accelerator->dmaHardware,
                                                 DMA_IRQ_ALL,
                                                 HARDWARE_TO_MEMORY);

    status = hardware_config->dmaDriver->InterruptSetHandler (accelerator->dmaHardware,
                                                              hardware_config->dmaRxIntVecID,
                                                              Accelerator_rxInterruptHandler,
                                                              accelerator);
    ASSERT(status == XST_SUCCESS);
    if (status != XST_SUCCESS) return status;
  }


  /***************************************************************************/
  /**************************** Accelerator initialization *******************/

  accelerator->updateHardware = hardware_config->hwDriver->new();

  ASSERT (accelerator->updateHardware != NULL);

  status = hardware_config->hwDriver->Initialize (accelerator->updateHardware,
                                                  hardware_config->hwDeviceID);
  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS) return XST_FAILURE;



  hardware_config->hwDriver->InterruptGlobalEnable (accelerator->updateHardware);
  hardware_config->hwDriver->InterruptEnable (accelerator->updateHardware, 1);

  status = hardware_config->hwDriver->InterruptSetHandler (accelerator->updateHardware,
                                                           hardware_config->hwIntVecID,
                                                           Accelerator_hardwareInterruptHandler,
                                                           accelerator);
  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS) return status;

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
      ARM_GIC_disconnect (accelerator->hardwareConfig->dmaTxIntVecID);

      ARM_GIC_disconnect (accelerator->hardwareConfig->dmaRxIntVecID);

      ARM_GIC_disconnect (accelerator->hardwareConfig->hwIntVecID);
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

void Accelerator_setup (SbSUpdateAccelerator * accelerator,
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

  {
    float values[1024];

    for (int i = 0; i < accelerator->profile->vectorSize; i++)
    {
#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))

      ((uint32_t*) values)[i] = DATA16_TO_FLOAT32(((uint16_t * ) state_vector)[i]);

      if (1.0 <= values[i])
      {
        printf ("Divergence");
      }
    }
  }

  if (accelerator->profile->vectorSize == 10)
  {
    *((float *) accelerator->txBufferCurrentPtr) = 0.0;
  }
  else
  {
    *((float *) accelerator->txBufferCurrentPtr) =
        ((float) MT19937_genrand ()) * (1.0 / (float) 0xFFFFFFFF);
  }

  accelerator->txBufferCurrentPtr += sizeof(float);

  memcpy (accelerator->txBufferCurrentPtr,
          state_vector,
          accelerator->profile->stateBufferSize);

  accelerator->txBufferCurrentPtr += accelerator->profile->stateBufferSize;

#ifdef DEBUG
  ASSERT(accelerator->txStateCounter <= accelerator->profile->layerSize);

  accelerator->txStateCounter ++;
#endif
}

inline void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          uint8_t * weight_vector) __attribute__((always_inline));

inline void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          uint8_t * weight_vector)
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

  memcpy (accelerator->txBufferCurrentPtr,
          weight_vector,
          accelerator->profile->weightBufferSize);

  accelerator->txBufferCurrentPtr += accelerator->profile->weightBufferSize;

#ifdef DEBUG
  accelerator->txWeightCounter ++;
#endif
}

typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

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

  if (accelerator->mode == SPIKE_MODE)
  {
    while (accelerator->acceleratorReady == 0);
    while (accelerator->txDone == 0);
    while (accelerator->rxDone == 0);

    accelerator->memory_cmd = accelerator->profile->memory_cmd[accelerator->mode];

    accelerator->acceleratorReady = 0;
    accelerator->hardwareConfig->hwDriver->Start (accelerator->updateHardware);


    accelerator->txDone = 0;
    status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                           accelerator->txBuffer,
                                                           accelerator->txBufferSize,
                                                           MEMORY_TO_HARDWARE);
    ASSERT(status == XST_SUCCESS);

    if (status == XST_SUCCESS)
    {
      accelerator->rxDone = 0;
      status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                             accelerator->rxBuffer,
                                                             accelerator->rxBufferSize,
                                                             HARDWARE_TO_MEMORY);

      ASSERT(status == XST_SUCCESS);
    }
  }
  else
  {
#define MAX_VECTOR_SIZE       (1024)
#define MAX_SPIKE_MATRIX_SIZE (60*60)
#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))
#define FLOAT32_TO_DATA16(d)  (0x0000FFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (0x000000FF & (((unsigned int)(d)) >> 19))
#define NEGLECTING_CONSTANT   ((float)1e-20)

    SbsAcceleratorProfie * profile = accelerator->profile;
    int index;
    float random_value;
    int last;
    Data32 register_A;
    Data32 register_B;
    uint32_t * stream_in = (uint32_t *) accelerator->txBuffer;
    uint32_t * stream_out = (uint32_t *) accelerator->rxBuffer;
    int i;

    int layerSize = profile->layerSize;
    int vectorSize = profile->vectorSize;
    int kernelSize = profile->kernelSize;
    float sum;
    unsigned short spikeID;
    int batch;
    unsigned int temp;
    unsigned int index_channel;

    float epsion_over_sum;
    float epsilon = *((float*) (&profile->epsilon));
    float reverse_epsilon = 1.0f / (1.0f + epsilon);

    float state_vector[MAX_VECTOR_SIZE];
    unsigned short spike_matrix[MAX_SPIKE_MATRIX_SIZE];
    float weight_vector[MAX_VECTOR_SIZE];
    float temp_data[MAX_VECTOR_SIZE];

    temp = 0;
    index_channel = 0;

    for (int ip_index = 0; ip_index < layerSize; ip_index++)
    {
      register_A.u32 = *(stream_in++);
      random_value = register_A.f32;
      ASSERT (random_value <= 1.0);

      index = 0;
      while (index < vectorSize)
      {
        i = *(stream_in++);

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
        index = 0;
        while (index < vectorSize)
        {
          i = *(stream_in++);

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
          temp_data[i] = state_vector[i] * weight_vector[i];
          sum += temp_data[i];
        }

        if (NEGLECTING_CONSTANT < sum)
        {
          epsion_over_sum = epsilon / sum;
          for (i = 0; i < vectorSize; i++)
          {
            state_vector[i] = reverse_epsilon
                * (state_vector[i] + temp_data[i] * epsion_over_sum);
          }
        }
      }

      for (i = 0; i < vectorSize; i++)
      {
        register_A.f32 = state_vector[i];

        if (1.0 <= register_A.f32)
        {
          printf("Divergence");
        }

        if ((register_A.u32 & 0xf0000000) == 0x30000000)
        {
          temp |= (FLOAT32_TO_DATA16(register_A.u32)) << (16 * index_channel);
        }

        index_channel ++;

        if (index_channel == 2)
        {
          *(stream_out ++) = temp;
          index_channel = 0;
          temp = 0;
        }
      }
    }

    index_channel = 0;
    temp = 0;
    for (i = 0; i < layerSize; i++)
    {
      temp |= ((unsigned int)spike_matrix[i]) << (16 * index_channel);
      index_channel ++;

      if ((index_channel == 2) || (i == layerSize - 1))
      {
        if (i == layerSize - 1)
        {
          last = 1;
          status = 0;
        }
        *(stream_out++) = temp;
        index_channel = 0;
        temp = 0;
      }
    }
  }

  return status;
}

/*****************************************************************************/

Result SbsPlatform_initialize (SbSHardwareConfig * hardware_config_list,
                               uint32_t list_length,
                               uint32_t MT19937_seed)
{
  int i;
  Result rc;

  rc = ARM_GIC_initialize ();

  ASSERT (rc == OK);

  if (rc != OK)
    return rc;

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

  if (MT19937_seed)
    MT19937_sgenrand (MT19937_seed);

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

