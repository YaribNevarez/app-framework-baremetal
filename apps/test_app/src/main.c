#include "stdio.h"
#include "test_app.h"
#include "test_platform.h"

static HardwareParameters hardwareParameters =
{
    .kernel =
    {
        .deviceId =     KERNEL_DEVICE_ID,
        .interruptId =  KERNEL_INTR_ID
    },
    .dma =
    {
        .deviceId =      DMA_DEVICE_ID,
        .txInterruptId = DMA_TX_INTR_ID,
        .rxInterruptId = DMA_RX_INTR_ID
    },
    .data =
    {
        .dataSize =         DATA_SIZE,
        .bufferLength =     BUFFER_LENGTH,
        .bufferInAddress =  BUFFER_IN_ADDRESS,
        .bufferOutAddress = BUFFER_OUT_ADDRESS
    }
};

int main (void)
{
  Result rc;

  printf ("==========  Xilinx Zynq 7000  ===============\n");
  printf ("==========  Memory bandwidth test ===========\n");

  TestApp * app = TestApp_instance ();

  rc = (app != NULL) ? OK : ERROR;

  if (rc == OK)
  {
    rc = app->initialize (app, &hardwareParameters);

    for (TestCase testCase = STREAM_DIRECT;
        testCase <= STREAM_FLUSH_PIPELINED;
        testCase++)
      if (rc == OK)
      {
        rc = app->run (app, testCase);
      }

    app->dispose (app);
  }

  return rc;
}
