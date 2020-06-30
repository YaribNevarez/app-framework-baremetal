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

// IFNDEF ----------------------------------------------------------------------
#ifndef TEST_APP_H_
#define TEST_APP_H_

#ifdef __cplusplus
extern "C" {
#endif

// INCLUDES --------------------------------------------------------------------
//#include "xil_types.h"
#include "stdint.h"
#include "stddef.h"

#include "test_app_config.h"

#include "result.h"
// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

typedef enum
{
  MASTER_DIRECT,
  MASTER_DIRECT_PIPELINE,
  MASTER_CACHED,
  MASTER_CACHED_PIPELINED,
  MASTER_CACHED_BURST,
  MASTER_SEND_BURST,
  MASTER_RETRIEVE_BURST,
  MASTER_SEND,
  MASTER_RETRIEVE,
  MASTER_SEND_PIPELINED,
  MASTER_RETRIEVE_PIPELINED,
  STREAM_DIRECT,
  STREAM_DIRECT_PIPELINED,
  STREAM_CACHED,
  STREAM_CACHED_PIPELINED,
  STREAM_SEND,
  STREAM_RETRIEVE,
  STREAM_SEND_PIPELINED,
  STREAM_RETRIEVE_PIPELINED
} TestCase;

typedef struct
{
  uint32_t deviceId;
  uint32_t txInterruptId;
  uint32_t rxInterruptId;
} DMAHwParameters;

typedef struct
{
  uint32_t deviceId;
  uint32_t interruptId;
} KernelHwParameters;

typedef struct
{
  uint32_t dataSize;
  uint32_t bufferLength;
  uint32_t maxBufferSize;
  void *   bufferInAddress;
  void *   bufferOutAddress;
} DataParameters;

typedef struct
{
  KernelHwParameters  kernel;
  DMAHwParameters     dma;
  DataParameters      data;
} HardwareParameters;

typedef struct TestApp_ TestApp;

struct TestApp_
{
  Result  (* initialize) (TestApp * self, HardwareParameters * hwParameters);
  Result  (* run)        (TestApp * self, TestCase testCase);
  void    (* dispose)    (TestApp * self);
};

// EUNUMERATIONS ---------------------------------------------------------------

// DECLARATIONS ----------------------------------------------------------------

TestApp * TestApp_instance(void);

char * TestCaseString_str (TestCase id);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* TEST_APP_H_ */
