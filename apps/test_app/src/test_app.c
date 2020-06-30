//------------------------------------------------------------------------------
/**
 *
 * @file: test_app.c
 *
 * @Created on: Jun 23rd, 2019
 * @Author: Yarib Nevarez
 *
 *
 * @brief - Memory bandwidth test application
 * <Requirement Doc Reference>
 * <Design Doc Reference>
 *
 * @copyright Free open source
 * 
 * yarib_007@hotmail.com
 * 
 * www.linkedin.com/in/yarib-nevarez
 *
 *
 */
//------------------------------------------------------------------------------
// INCLUDES --------------------------------------------------------------------
#include "test_app.h"
#include "stdio.h"

#include "xstatus.h"
#include "ff.h"

#include "eventlogger.h"
#include "test_platform.h"
#include "toolcom.h"
#include "miscellaneous.h"

#include "dma_hardware_mover.h"

#include "xtest_module.h"

// FORWARD DECLARATIONS --------------------------------------------------------

// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

// EUNUMERATIONS ---------------------------------------------------------------

typedef struct
{
  TestCase  id;
  char      str[32];
} TestCaseString;

static TestCaseString TestCaseString_data[] =
{
  { MASTER_DIRECT,              "MASTER_DIRECT"             },
  { MASTER_DIRECT_PIPELINE,     "MASTER_DIRECT_PIPELINE"    },
  { MASTER_CACHED,              "MASTER_CACHED"             },
  { MASTER_CACHED_PIPELINED,    "MASTER_CACHED_PIPELINED"   },
  { MASTER_CACHED_BURST,        "MASTER_CACHED_BURST"       },
  { MASTER_SEND_BURST,          "MASTER_SEND_BURST"         },
  { MASTER_RETRIEVE_BURST,      "MASTER_RETRIEVE_BURST"     },
  { MASTER_SEND,                "MASTER_SEND"               },
  { MASTER_RETRIEVE,            "MASTER_RETRIEVE"           },
  { MASTER_SEND_PIPELINED,      "MASTER_SEND_PIPELINED"     },
  { MASTER_RETRIEVE_PIPELINED,  "MASTER_RETRIEVE_PIPELINED" },
  { STREAM_DIRECT,              "STREAM_DIRECT"             },
  { STREAM_DIRECT_PIPELINED,    "STREAM_DIRECT_PIPELINED"   },
  { STREAM_CACHED,              "STREAM_CACHED"             },
  { STREAM_CACHED_PIPELINED,    "STREAM_CACHED_PIPELINED"   },
  { STREAM_SEND,                "STREAM_SEND"               },
  { STREAM_RETRIEVE,            "STREAM_RETRIEVE"           },
  { STREAM_SEND_PIPELINED,      "STREAM_SEND_PIPELINED"     },
  { STREAM_RETRIEVE_PIPELINED,  "STREAM_RETRIEVE_PIPELINED" },
  { 0 }
};

char * TestCaseString_str (TestCase id)
{
  char * str = NULL;

  for (uint32_t i = 0;
      (i < sizeof(TestCaseString_data) / sizeof(TestCaseString)) && str == NULL;
      i++)
    if (TestCaseString_data[i].id == id)
      str = TestCaseString_data[i].str;

  return str ? str : "NONE";
}
// STRUCTS AND NAMESPACES ------------------------------------------------------

typedef struct
{
  void *    (*new)(void);
  void      (*delete)(void ** InstancePtr);

  int       (*Initialize) (void *InstancePtr, uint16_t deviceId);
  void      (*Start)      (void *InstancePtr);
  uint32_t  (*IsDone)     (void *InstancePtr);
  uint32_t  (*IsIdle)     (void *InstancePtr);
  uint32_t  (*IsReady)    (void *InstancePtr);
  void      (*EnableAutoRestart)  (void *InstancePtr);
  void      (*DisableAutoRestart) (void *InstancePtr);
  uint32_t  (*Get_return) (void *InstancePtr);

  void      (*Set_test_case)     (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_test_case)     (void *InstancePtr);
  void      (*Set_buffer_length) (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_buffer_length) (void *InstancePtr);
  void      (*Set_master_in)     (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_master_in)     (void *InstancePtr);
  void      (*Set_master_out)    (void *InstancePtr, uint32_t Data);
  uint32_t  (*Get_master_out)    (void *InstancePtr);

  void      (*InterruptGlobalEnable)  (void *InstancePtr);
  void      (*InterruptGlobalDisable) (void *InstancePtr);
  void      (*InterruptEnable)        (void *InstancePtr, uint32_t Mask);
  void      (*InterruptDisable)       (void *InstancePtr, uint32_t Mask);
  void      (*InterruptClear)         (void *InstancePtr, uint32_t Mask);
  uint32_t  (*InterruptGetEnabled)    (void *InstancePtr);
  uint32_t  (*InterruptGetStatus)     (void *InstancePtr);

  uint32_t  (*InterruptSetHandler)    (void *InstancePtr,
                                       uint32_t ID,
                                       ARM_GIC_InterruptHandler handler,
                                       void * data);
} HWKernel;

static void * HWKernel_new (void)
{
  return malloc (sizeof(XTest_module));
}

static void HWKernel_delete (void ** InstancePtr)
{
  if (InstancePtr && *InstancePtr)
  {
    free (*InstancePtr);
    *InstancePtr = NULL;
  }
}

static uint32_t  HWKernel_InterruptSetHandler (void *instance,
                                               uint32_t ID,
                                               ARM_GIC_InterruptHandler handler,
                                               void * data)
{
  uint32_t status = ARM_GIC_connect (ID, handler, data);
  ASSERT (status == XST_SUCCESS);
  return status;
}

HWKernel HWKernel_driver =
{
  .new =                HWKernel_new,
  .delete =             HWKernel_delete,
  .Initialize =         (int (*)(void *, uint16_t)) XTest_module_Initialize,
  .Start =              (void (*)(void *)) XTest_module_Start,
  .IsDone =             (uint32_t(*)(void *)) XTest_module_IsDone,
  .IsIdle =             (uint32_t(*) (void *)) XTest_module_IsIdle,
  .IsReady =            (uint32_t(*) (void *)) XTest_module_IsReady,
  .EnableAutoRestart =  (void (*) (void *)) XTest_module_EnableAutoRestart,
  .DisableAutoRestart = (void (*) (void *)) XTest_module_DisableAutoRestart,
  .Get_return =         (uint32_t(*) (void *))  XTest_module_Get_return,

  .Set_test_case =      (void (*) (void *, uint32_t )) XTest_module_Set_test_case,
  .Get_test_case =      (uint32_t(*) (void *)) XTest_module_Get_test_case,
  .Set_buffer_length =  (void (*) (void *, uint32_t )) XTest_module_Set_buffer_length,
  .Get_buffer_length =  (uint32_t(*) (void *)) XTest_module_Get_buffer_length,
  .Set_master_in =      (void (*) (void *, uint32_t )) XTest_module_Set_master_in_V,
  .Get_master_in =      (uint32_t(*) (void *)) XTest_module_Get_master_in_V,
  .Set_master_out =     (void (*) (void *, uint32_t )) XTest_module_Set_master_out_V,
  .Get_master_out =     (uint32_t(*) (void *)) XTest_module_Get_master_out_V,

  .InterruptGlobalEnable =  (void (*) (void *)) XTest_module_InterruptGlobalEnable,
  .InterruptGlobalDisable = (void (*) (void *)) XTest_module_InterruptGlobalDisable,
  .InterruptEnable =        (void (*) (void *, uint32_t )) XTest_module_InterruptEnable,
  .InterruptDisable =       (void (*) (void *, uint32_t )) XTest_module_InterruptDisable,
  .InterruptClear =         (void (*) (void *, uint32_t )) XTest_module_InterruptClear,
  .InterruptGetEnabled =    (uint32_t(*) (void *)) XTest_module_InterruptGetEnabled,
  .InterruptGetStatus =     (uint32_t(*) (void *)) XTest_module_InterruptGetStatus,

  .InterruptSetHandler =    HWKernel_InterruptSetHandler
};

typedef struct
{
  // Object virtual function table
  TestApp       virtualTable;

  // File system
  FATFS         fatFS;

  // Hardware parameters
  HardwareParameters * hwParameters;

  // DMA members
  DMAHardware * dmaDriver;
  void *        dma;
  uint32_t      dmaTxDone;
  uint32_t      dmaRxDone;

  // Hardware members
  void *        hw;
  HWKernel *    hwDriver;
  uint32_t      hwDone;

  // Application tools
  Timer *       timer;
  uint32_t      errorFlags;
  double        dmaTxTime;
  double        dmaRxTime;
  double        hwTime;
} TestAppPrivate;

// DEFINITIONs -----------------------------------------------------------------

static uint32_t TestApp_initializeSDCard (TestAppPrivate * self)
{
  FRESULT rc;
  TCHAR *path = "0:/"; /* Logical drive number is 0 */

  /* Register volume work area, initialize device */
  rc = f_mount (&self->fatFS, path, 0);

  ASSERT (rc == FR_OK);

  if (rc != FR_OK)
    return ERROR;


  return OK;
}


#define DMA_RESET_TIMEOUT 10000

static void TestApp_txInterruptHandler (void * obj)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  DMAIRQMask irq_status = self->dmaDriver->InterruptGetStatus (self->dma, MEMORY_TO_HARDWARE);

  self->dmaDriver->InterruptClear (self->dma, irq_status, MEMORY_TO_HARDWARE);

   if (!(irq_status & DMA_IRQ_ALL)) return;

   if (irq_status & DMA_IRQ_DELAY) return;

   if (irq_status & DMA_IRQ_ERROR)
   {
     int TimeOut;

     self->errorFlags |= 0x01;

     self->dmaDriver->Reset (self->dma);

    for (TimeOut = DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (self->dmaDriver->ResetIsDone (self->dma))
        break;

     ASSERT(0);
     return;
   }

   if (irq_status & DMA_IRQ_IOC)
   {
     self->dmaTxDone = 1;
     self->dmaTxTime = Timer_getCurrentTime (self->timer);
   }
}

static void TestApp_rxInterruptHandler(void * obj)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  DMAIRQMask irq_status = self->dmaDriver->InterruptGetStatus (self->dma, HARDWARE_TO_MEMORY);

  self->dmaDriver->InterruptClear (self->dma, irq_status, HARDWARE_TO_MEMORY);

   if (!(irq_status & DMA_IRQ_ALL)) return;

   if (irq_status & DMA_IRQ_DELAY) return;

   if (irq_status & DMA_IRQ_ERROR)
   {
     int TimeOut;

     self->errorFlags |= 0x01;

     self->dmaDriver->Reset (self->dma);

    for (TimeOut = DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (self->dmaDriver->ResetIsDone (self->dma))
        break;

     ASSERT(0);
     return;
   }

   if (irq_status & DMA_IRQ_IOC)
   {
     self->dmaRxDone = 1;
     self->dmaRxTime = Timer_getCurrentTime (self->timer);;
   }
}

static void TestApp_hardwareInterruptHandler (void * obj)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  uint32_t status;

  status = self->hwDriver->InterruptGetStatus (self->hw);
  self->hwDriver->InterruptClear (self->hw, status);

  self->hwDone = status & 1;
  self->hwTime = Timer_getCurrentTime (self->timer);
}

static uint32_t TestApp_initializeDMA (TestAppPrivate * self)
{
  uint32_t status = OK;

  ASSERT (self != NULL);

  if (self == NULL)
    return ERROR;

  ASSERT (self->hwParameters != NULL);

  if (self->hwParameters == NULL)
    return ERROR;

  ASSERT (self->dmaDriver != NULL);

  if (self->dmaDriver == NULL)
    return ERROR;

  self->dma = self->dmaDriver->new ();

  ASSERT (self->dma != NULL);

  if (self->dma == NULL)
    return ERROR;

  status = self->dmaDriver->Initialize (self->dma, self->hwParameters->dma.deviceId);

  ASSERT (status == OK);

  if (status != OK)
    return ERROR;

  if (self->hwParameters->dma.txInterruptId != 0)
  {
    self->dmaDriver->InterruptEnable (self->dma, DMA_IRQ_ALL, MEMORY_TO_HARDWARE);

    status = self->dmaDriver->InterruptSetHandler (self->dma,
                                                   self->hwParameters->dma.txInterruptId,
                                                   TestApp_txInterruptHandler,
                                                   self);
    ASSERT(status == OK);
    if (status != OK)
      return status;
  }

  if (self->hwParameters->dma.rxInterruptId != 0)
  {
    self->dmaDriver->InterruptEnable (self->dma,
                                      DMA_IRQ_ALL,
                                      HARDWARE_TO_MEMORY);

    status = self->dmaDriver->InterruptSetHandler (self->dma,
                                                   self->hwParameters->dma.rxInterruptId,
                                                   TestApp_rxInterruptHandler,
                                                   self);
    ASSERT(status == OK);
    if (status != OK)
      return status;
  }

  return OK;
}

static uint32_t TestApp_initializeHWKernel (TestAppPrivate * self)
{
  uint32_t status = OK;

  ASSERT (self != NULL);

   if (self == NULL)
     return ERROR;

  ASSERT (self->hwParameters != NULL);

  if (self->hwParameters == NULL)
    return ERROR;

  self->hw = self->hwDriver->new ();

  ASSERT (self->hw != NULL);

  status = self->hwDriver->Initialize (self->hw, self->hwParameters->kernel.deviceId);
  ASSERT(status == OK);

  if (status != OK)
    return ERROR;

  if (self->hwParameters->kernel.interruptId)
  {
    self->hwDriver->InterruptGlobalEnable (self->hw);
    self->hwDriver->InterruptEnable (self->hw, 1);

    status = self->hwDriver->InterruptSetHandler (self->hw,
                                                  self->hwParameters->kernel.interruptId,
                                                  TestApp_hardwareInterruptHandler,
                                                  self);
  }

  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS)
    return status;

  return OK;
}

static Result TestApp_initialize (TestApp * obj, HardwareParameters * hwParameters)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  Result rc;

  ASSERT (hwParameters != NULL)

  if (hwParameters == NULL)
    return ERROR;

  self->hwParameters = hwParameters;

  rc = ARM_GIC_initialize ();

  if (rc != OK)
  {
    printf ("ARM GIC initialize error\n");
    return rc;
  }

  rc = TestApp_initializeSDCard (self);

  if (rc != OK)
  {
    printf ("SD card hardware error\n");
    return rc;
  }

  rc = TestApp_initializeDMA (self);

  if (rc != OK)
  {
    printf ("DMA hardware initialization error\n");
    return rc;
  }

  rc = TestApp_initializeHWKernel (self);

  if (rc != OK)
  {
    printf ("Kernel hardware initialization error\n");
    return rc;
  }

  return rc;
}

static Result TestApp_run (TestApp * obj, TestCase test_case)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  uint32_t matching_flag;

  float currentTime = 0;

  float bandwidth = 0;
  float txBandwidth = 0;
  float rxBandwidth = 0;

  float txLatency = 0;
  float rxLatency = 0;

  void * bufferIn = self->hwParameters->data.bufferInAddress;
  void * bufferOut = self->hwParameters->data.bufferOutAddress;
  size_t bufferSize = self->hwParameters->data.bufferLength * self->hwParameters->data.dataSize;
  size_t bufferLength = self->hwParameters->data.bufferLength;

  printf ("\nLength = %d, wide = %d-Bit, buffer size = %d-Byte\n",
          (int) self->hwParameters->data.bufferLength,
          (int )self->hwParameters->data.dataSize * 8,
          bufferSize);

  self->dmaRxDone = 0;
  self->dmaTxDone = 0;
  self->hwDone = 0;

  self->dmaRxTime = 0.0;
  self->dmaTxTime = 0.0;
  self->hwTime = 0.0;

  memset (bufferIn, 0x00, self->hwParameters->data.maxBufferSize);
  memset (bufferOut, 0x00, self->hwParameters->data.maxBufferSize);

  Xil_DCacheFlushRange ((INTPTR) bufferIn,  self->hwParameters->data.maxBufferSize);
  Xil_DCacheFlushRange ((INTPTR) bufferOut, self->hwParameters->data.maxBufferSize);

  self->hwDriver->Set_buffer_length (self->hw, bufferLength);
  self->hwDriver->Set_test_case (self->hw, test_case);
  self->hwDriver->Set_master_in (self->hw, (uint32_t) bufferIn);
  self->hwDriver->Set_master_out (self->hw, (uint32_t) bufferOut);

  //memset (bufferIn,  0xA5, bufferSize);
  for (uint32_t * ptr = (uint32_t *) bufferIn; ptr < (bufferIn + bufferSize); ptr++)
  {
    *ptr = (uint32_t) ptr;
  }

  Xil_DCacheFlushRange ((INTPTR) bufferIn,  bufferSize);

  Timer_start (self->timer);

  switch(test_case)
  {
    case MASTER_DIRECT:
    case MASTER_DIRECT_PIPELINE:
    case MASTER_CACHED:
    case MASTER_CACHED_PIPELINED:
    case MASTER_CACHED_BURST:
    case MASTER_SEND_BURST:
    case MASTER_RETRIEVE_BURST:
    case MASTER_SEND:
    case MASTER_RETRIEVE:
    case MASTER_SEND_PIPELINED:
    case MASTER_RETRIEVE_PIPELINED:

      self->hwDriver->Start (self->hw);
      while (!self->hwDone && currentTime < 1.0)
        currentTime = Timer_getCurrentTime (self->timer);

      break;
    case STREAM_DIRECT:
    case STREAM_DIRECT_PIPELINED:
    case STREAM_CACHED:
    case STREAM_CACHED_PIPELINED:
    case STREAM_SEND:
    case STREAM_RETRIEVE:
    case STREAM_SEND_PIPELINED:
    case STREAM_RETRIEVE_PIPELINED:

      self->hwDriver->Start (self->hw);

      if(test_case != STREAM_RETRIEVE && test_case != STREAM_RETRIEVE_PIPELINED)
        self->dmaDriver->Move (self->dma, bufferIn, bufferSize, MEMORY_TO_HARDWARE);

      if(test_case != STREAM_SEND && test_case != STREAM_SEND_PIPELINED)
        self->dmaDriver->Move (self->dma, bufferOut, bufferSize, HARDWARE_TO_MEMORY);

      while (!self->dmaTxDone
          && test_case != STREAM_RETRIEVE
          && test_case != STREAM_RETRIEVE_PIPELINED
          && currentTime < 1.0)
        currentTime = Timer_getCurrentTime (self->timer);

      while (!self->dmaRxDone
          && test_case != STREAM_SEND
          && test_case != STREAM_SEND_PIPELINED
          && currentTime < 1.0)
        currentTime = Timer_getCurrentTime (self->timer);

      while (!self->hwDone && currentTime < 1.0)
        currentTime = Timer_getCurrentTime (self->timer);

      break;
    default:;
  }

  switch(test_case)
  {
    case MASTER_DIRECT:
    case MASTER_DIRECT_PIPELINE:
      bandwidth = bufferSize / self->hwTime;
      break;
    case MASTER_CACHED:
    case MASTER_CACHED_PIPELINED:
    case MASTER_CACHED_BURST:
      bandwidth = (2 * bufferSize) / self->hwTime;
      break;
    case MASTER_SEND_BURST:
    case MASTER_RETRIEVE_BURST:
    case MASTER_SEND:
    case MASTER_RETRIEVE:
    case MASTER_SEND_PIPELINED:
    case MASTER_RETRIEVE_PIPELINED:
      bandwidth = bufferSize / self->hwTime;
      break;
    case STREAM_DIRECT:
    case STREAM_DIRECT_PIPELINED:
      txBandwidth = bufferSize / self->dmaTxTime;
      txLatency = self->dmaTxTime;
      break;
    case STREAM_CACHED:
    case STREAM_CACHED_PIPELINED:
      txBandwidth = bufferSize / self->dmaTxTime;
      rxBandwidth = bufferSize / (self->dmaRxTime - self->dmaTxTime);

      txLatency = self->dmaTxTime;
      rxLatency = self->dmaRxTime - self->dmaTxTime;
      break;
    case STREAM_SEND:
      txBandwidth = bufferSize / self->dmaTxTime;
      txLatency = self->dmaTxTime;
      break;
    case STREAM_RETRIEVE:
      rxBandwidth = bufferSize / self->dmaRxTime;
      rxLatency = self->dmaRxTime;
      break;
    case STREAM_SEND_PIPELINED:
      txBandwidth = bufferSize / self->dmaTxTime;
      txLatency = self->dmaTxTime;
      break;
    case STREAM_RETRIEVE_PIPELINED:
      rxBandwidth = bufferSize / self->dmaRxTime;
      rxLatency = self->dmaRxTime;
      break;
    default:
      ASSERT (0);
  }


  if (0.0 < txBandwidth)
  {
    printf ("DMA Tx bandwidth = %f Mb/S, Tx latency = %.3f uS\n", txBandwidth / (1024 * 1024), 1e6 * txLatency);
  }

  if (0.0 < rxBandwidth)
  {
    printf ("DMA Rx bandwidth = %f Mb/S, Rx latency = %.3f uS\n", rxBandwidth / (1024 * 1024), 1e6 * rxLatency);
  }

  if (0.0 < bandwidth)
  {
    printf ("Kernel Hardware bandwidth = %f Mb/S, HW latency = %.3f uS\n", bandwidth / (1024 * 1024), 1e6 * self->hwTime);
  }


  Xil_DCacheInvalidateRange ((INTPTR) bufferOut, self->hwParameters->data.maxBufferSize);

  if (   test_case != STREAM_SEND
      && test_case != STREAM_SEND_PIPELINED
      && test_case != MASTER_SEND
      && test_case != MASTER_SEND_PIPELINED
      && test_case != MASTER_SEND_BURST)
  {
    matching_flag = !memcmp (bufferIn, bufferOut, bufferSize);
    printf ("Inside memory region: %s\n", matching_flag ? "PASS" : "FAIL");

    matching_flag = !memcmp (bufferIn, bufferOut, self->hwParameters->data.maxBufferSize);
    printf ("Outside memory region: %s\n", matching_flag ? "PASS" : "FAIL");
  }

  if (currentTime >= 1.0)
    printf ("TIMEOUT\n");

  return OK;
}

static void TestApp_dispose (TestApp * obj)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;

  Timer_delete (&self->timer);

  self->dmaDriver->delete (&self->dma);

  self->hwDriver->delete (&self->hw);
}

TestApp TestApp_vtbl = { TestApp_initialize, TestApp_run, TestApp_dispose };

TestApp * TestApp_instance (void)
{
  static TestAppPrivate TestApp_obj = { 0 };


  TestApp_obj.virtualTable = TestApp_vtbl;

  TestApp_obj.dmaDriver = &DMAHardware_mover;
  TestApp_obj.hwDriver = &HWKernel_driver;

  TestApp_obj.timer = Timer_new (1);

  ASSERT (TestApp_obj.timer != NULL);


  return (TestApp *) &TestApp_obj;
}
