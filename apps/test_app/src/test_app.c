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
typedef enum
{
  MASTER_DIRECT,
  MASTER_DIRECT_PIPELINE,
  MASTER_CACHED,
  MASTER_CACHED_PIPELINED,
  MASTER_CACHED_BURST,
  MASTER_STORE,
  MASTER_FLUSH,
  MASTER_STORE_PIPELINED,
  MASTER_FLUSH_PIPELINED,
  STREAM_DIRECT,
  STREAM_DIRECT_PIPELINED,
  STREAM_CACHED,
  STREAM_CACHED_PIPELINED,
  STREAM_STORE,
  STREAM_FLUSH,
  STREAM_STORE_PIPELINED,
  STREAM_FLUSH_PIPELINED
} TestCase;
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

  void *        bufferIn;
  uint32_t      bufferInLength;
  void *        bufferOut;
  uint32_t      bufferOutLength;
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

void TestApp_txInterruptHandler(void * obj)
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
     self->dmaTxTime = Timer_getCurrentTime (self->timer);;
   }
}

void TestApp_rxInterruptHandler(void * obj)
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

void TestApp_hardwareInterruptHandler (void * obj)
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
  uint16_t dmaDeviceID = XPAR_AXI_DMA_0_DEVICE_ID;
  uint16_t dmaTxIntVecID = XPAR_FABRIC_AXI_DMA_0_MM2S_INTROUT_INTR;
  uint16_t dmaRxIntVecID = XPAR_FABRIC_AXI_DMA_0_S2MM_INTROUT_INTR;

  ASSERT (self->dmaDriver != NULL);

  if (self->dmaDriver == NULL)
    return ERROR;

  self->dma = self->dmaDriver->new ();

  ASSERT (self->dma != NULL);

  if (self->dma == NULL)
    return ERROR;

  status = self->dmaDriver->Initialize (self->dma, dmaDeviceID);

  ASSERT (status == OK);

  if (status != OK)
    return ERROR;

  if (dmaTxIntVecID != 0)
  {
    self->dmaDriver->InterruptEnable (self->dma, DMA_IRQ_ALL, MEMORY_TO_HARDWARE);

    status = self->dmaDriver->InterruptSetHandler (self->dma,
                                                   dmaTxIntVecID,
                                                   TestApp_txInterruptHandler,
                                                   self);
    ASSERT(status == OK);
    if (status != OK)
      return status;
  }

  if (dmaRxIntVecID != 0)
  {
    self->dmaDriver->InterruptEnable (self->dma,
                                      DMA_IRQ_ALL,
                                      HARDWARE_TO_MEMORY);

    status = self->dmaDriver->InterruptSetHandler (self->dma,
                                                   dmaRxIntVecID,
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
  uint16_t hwDeviceID = XPAR_TEST_MODULE_0_DEVICE_ID;
  uint16_t hwIntVecID = XPAR_FABRIC_TEST_MODULE_0_INTERRUPT_INTR;

  self->hw = self->hwDriver->new ();

  ASSERT (self->hw != NULL);

  status = self->hwDriver->Initialize (self->hw, hwDeviceID);
  ASSERT(status == OK);

  if (status != OK)
    return ERROR;

  if (hwIntVecID)
  {
    self->hwDriver->InterruptGlobalEnable (self->hw);
    self->hwDriver->InterruptEnable (self->hw, 1);

    status = self->hwDriver->InterruptSetHandler (self->hw,
                                                  hwIntVecID,
                                                  TestApp_hardwareInterruptHandler,
                                                  self);
  }

  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS)
    return status;

  return OK;
}

Result TestApp_initialize (TestApp * obj)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  Result rc;

  uint32_t bufferInLength = 1024 * sizeof(uint32_t);
  uint32_t bufferOutLength = 1024 * sizeof(uint32_t);
  void * bufferIn = (void *) (XPAR_PS7_DDR_0_S_AXI_HIGHADDR - 1024*1024 + 1);
  void * bufferOut = (void *) (XPAR_PS7_DDR_0_S_AXI_HIGHADDR - 512*1024 + 1);

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


  self->bufferIn = bufferIn;
  self->bufferInLength = bufferInLength;

  self->bufferOut = bufferOut;
  self->bufferOutLength = bufferOutLength;

  return rc;
}

Result TestApp_run (TestApp * obj)
{
  TestAppPrivate * self = (TestAppPrivate *) obj;
  uint32_t correct_flag;
  TestCase test_case = STREAM_CACHED_PIPELINED;

  // ********** Create SBS Neural Network **********
  printf ("\n==========  Xilinx Zynq 7000  ===============");
  printf ("\n==========  Memory bandwidth test ===========");

  for (uint8_t * ptr = self->bufferIn;
      (void *) ptr < (self->bufferIn + self->bufferInLength);
      ptr ++)
    *ptr = ((int)ptr) % 9 + 1;

  ((uint32_t*) self->bufferIn)[self->bufferInLength / sizeof(uint32_t) - 1] = 0xFFFFFFFF;

  memset (self->bufferOut, 0, self->bufferOutLength);

  self->dmaRxDone = 0;
  self->dmaTxDone = 0;
  self->hwDone = 0;

  self->dmaRxTime = 0.0;
  self->dmaTxTime = 0.0;
  self->hwTime = 0.0;

  self->hwDriver->Set_buffer_length (self->hw, (uint32_t) self->bufferOutLength / sizeof(uint32_t));
  self->hwDriver->Set_test_case (self->hw, test_case);

  Xil_DCacheFlushRange ((INTPTR) self->bufferIn, self->bufferInLength);

  Timer_start (self->timer);

  switch(test_case)
  {
    case MASTER_DIRECT:
    case MASTER_DIRECT_PIPELINE:
    case MASTER_CACHED:
    case MASTER_CACHED_PIPELINED:
    case MASTER_CACHED_BURST:
    case MASTER_STORE:
    case MASTER_FLUSH:
    case MASTER_STORE_PIPELINED:
    case MASTER_FLUSH_PIPELINED:

      self->hwDriver->Set_master_in (self->hw, (uint32_t) self->bufferIn);
      self->hwDriver->Set_master_out (self->hw, (uint32_t) self->bufferOut);
      self->hwDriver->Start (self->hw);
      while (!self->hwDone);

      break;
    case STREAM_DIRECT:
    case STREAM_DIRECT_PIPELINED:
    case STREAM_CACHED:
    case STREAM_CACHED_PIPELINED:
    case STREAM_STORE:
    case STREAM_FLUSH:
    case STREAM_STORE_PIPELINED:
    case STREAM_FLUSH_PIPELINED:

      self->hwDriver->Start (self->hw);

      self->dmaDriver->Move(self->dma,
                            self->bufferIn,
                            self->bufferInLength,
                            MEMORY_TO_HARDWARE);

      self->dmaDriver->Move(self->dma,
                            self->bufferOut,
                            self->bufferOutLength,
                            HARDWARE_TO_MEMORY);

      while (!self->dmaTxDone);
      while (!self->dmaRxDone);
      while (!self->hwDone);

      printf ("\ndmaTxTime = %.16f", self->dmaTxTime);
      printf ("\ndmaRxTime = %.16f", self->dmaRxTime);

      break;
    default:;
  }

  Xil_DCacheInvalidateRange ((INTPTR) self->bufferOut, self->bufferOutLength);

  correct_flag = !memcmp (self->bufferIn, self->bufferOut, self->bufferOutLength);

  printf ("\nhwTime = %.16f", self->hwTime);
  printf ("\n%s\n", correct_flag ? "PASS" : "FAIL");

  return OK;
}

void TestApp_dispose (TestApp * obj)
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
