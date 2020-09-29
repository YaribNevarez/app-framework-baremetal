//------------------------------------------------------------------------------
/**
 *
 * @file: sbs_app.c
 *
 * @Created on: September 9th, 2019
 * @Author: Yarib Nevarez
 *
 *
 * @brief - Spike by Spike Neural Network test application
 * <Requirement Doc Reference>
 * <Design Doc Reference>
 *
 * @copyright Copyright [2019] FREE
 * All Rights Reserved.
 *
 *
 */
//------------------------------------------------------------------------------
// INCLUDES --------------------------------------------------------------------
#include "sbs_app.h"
#include "sbs_neural_network.h"
#include "stdio.h"

#include "xstatus.h"
#include "ff.h"

#include "eventlogger.h"
#include "sbs_processing_unit.h"
#include "sbs_platform.h"
#include "toolcom.h"

// INCLUDES --------------------------------------------------------------------
#include "sbs_processing_unit.h"
#include "sbs_hardware_spike.h"
#include "sbs_hardware_update.h"
#include "sbs_custom_hardware.h"
#include "sbs_pooling_layer.h"
#include "sbs_conv_layer_64.h"
#include "sbs_conv_layer_32.h"
#include "dma_hardware_mover.h"

// FORWARD DECLARATIONS --------------------------------------------------------

// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

// EUNUMERATIONS ---------------------------------------------------------------

// STRUCTS AND NAMESPACES ------------------------------------------------------

// DEFINITIONs -----------------------------------------------------------------

SbSHardwareConfig SbSHardwareConfig_list[] =
{
#ifdef HW_SPIKE_UNIT_0
  { .hwDriver      = &SbsHardware_fixedpoint_spike,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_SPIKE_UNIT_0,
    .hwDeviceID    = XPAR_SBS_SPIKE_UNIT_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_SPIKE_UNIT_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_SPIKE_UNIT_0_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x31FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_ACCELERATOR_UNIT_0
  { .hwDriver      = &SbsHardware_custom,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_ACCELERATOR_UNIT_0,
    .hwDeviceID    = XPAR_SBS_ACCELERATOR_UNIT_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_ACCELERATOR_UNIT_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_ACCELERATOR_UNIT_0_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x32000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x32FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_POOLING_LAYER_0
  { .hwDriver      = &SbsHardware_poolingLayer,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_POOLING_LAYER_0,
    .hwDeviceID    = XPAR_SBS_POOLING_LAYER_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_POOLING_LAYER_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_POOLING_LAYER_0_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x33000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x33FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_POOLING_LAYER_1
  { .hwDriver      = &SbsHardware_poolingLayer,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_POOLING_LAYER_1,
    .hwDeviceID    = XPAR_SBS_POOLING_LAYER_1_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_POOLING_LAYER_1_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_POOLING_LAYER_1_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x34000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x34FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_CONVOLUTION_LAYER_0
  { .hwDriver      = &SbsHardware_convLayer32,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_CONVOLUTION_LAYER_0,
    .hwDeviceID    = XPAR_SBS_CONV_LAYER_32_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_CONV_LAYER_32_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_CONV_LAYER_32_0_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x35000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x35FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_CONVOLUTION_LAYER_1
  { .hwDriver      = &SbsHardware_convLayer64,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_CONVOLUTION_LAYER_1,
    .hwDeviceID    = XPAR_SBS_CONV_LAYER_64_0_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_CONV_LAYER_64_0_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_CONV_LAYER_64_0_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x36000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x36FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_CONVOLUTION_LAYER_2
  { .hwDriver      = &SbsHardware_convLayer64,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_CONVOLUTION_LAYER_2,
    .hwDeviceID    = XPAR_SBS_CONV_LAYER_64_1_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_CONV_LAYER_64_1_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_CONV_LAYER_64_1_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x37000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x37FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
#ifdef HW_POOLING_LAYER_2
  { .hwDriver      = &SbsHardware_poolingLayer,
    .dmaDriver     = &DMAHardware_mover,
    .layerAssign   = HW_POOLING_LAYER_2,
    .hwDeviceID    = XPAR_SBS_POOLING_LAYER_2_DEVICE_ID,
    .dmaDeviceID   = XPAR_DMA_POOLING_LAYER_2_DEVICE_ID,
    .hwIntVecID    = XPAR_FABRIC_SBS_POOLING_LAYER_2_INTERRUPT_INTR,
    .dmaTxIntVecID = 0,
    .dmaRxIntVecID = 0,
    .channelSize   = 4,
    .ddrMem =
    { .baseAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x38000000,
      .highAddress = XPAR_PS7_DDR_0_S_AXI_BASEADDR + 0x38FFFFFF,
      .blockIndex  = 0
    }
  },
#endif
};


static FATFS fatfs;
static u32 SnnApp_initializeSD(void)
{

  FRESULT rc;
  TCHAR *path = "0:/"; /* Logical drive number is 0 */

  /* Register volume work area, initialize device */
  rc = f_mount (&fatfs, path, 0);

  if (rc != FR_OK)
  {
    return XST_FAILURE;
  }

  return OK;
}


Result SnnApp_initialize(void)
{
  Result rc;

  rc = SnnApp_initializeSD();

  if (rc != OK)
  {
    printf ("SD card hardware error\n");
    return rc;
  }

  rc = SbsPlatform_initialize (SbSHardwareConfig_list,
                               sizeof(SbSHardwareConfig_list) / sizeof(SbSHardwareConfig),
                               MT19937_SEED);

  if (rc != OK)
  {
    printf ("SbS hardware platform error\n");
    return rc;
  }

  return OK;
}

typedef struct
{
  int input_pattern_first;
  int input_pattern_last;
  int number_of_spikes;
  int result_correct_inferences;
  int result_total_inferences;
  float result_accuracy;
} SbsStatistics;

typedef struct
{
  float noise;
  char path[80];
  char label[80];
  SbsStatistics statistics_array[20];
} SbsBatchStatistics;

#if SBS_TAKE_ACCURACY_STATISTICS
static SbsStatistics statistics[] =
  {
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 200
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 300
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 400
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 500
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 600
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 700
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 800
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 900
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 1000
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 1100
    },
    { .input_pattern_first = SBS_INPUT_PATTERN_FIRST,
      .input_pattern_last = SBS_INPUT_PATTERN_LAST,
      .number_of_spikes = 1200
    },
  };

static SbsBatchStatistics batch_statistics[] =
{
    { .noise = 0,
      .path = "/MNIST/N_0/In%d.bin",
      .label = "0% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.05,
      .path = "/MNIST/N_5/In%d.bin",
      .label = "5% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.10,
      .path = "/MNIST/N_10/In%d.bin",
      .label = "10% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.15,
      .path = "/MNIST/N_15/In%d.bin",
      .label = "15% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.20,
      .path = "/MNIST/N_20/In%d.bin",
      .label = "20% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.25,
      .path = "/MNIST/N_25/In%d.bin",
      .label = "25% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.30,
      .path = "/MNIST/N_30/In%d.bin",
      .label = "30% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.35,
      .path = "/MNIST/N_35/In%d.bin",
      .label = "35% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.40,
      .path = "/MNIST/N_40/In%d.bin",
      .label = "40% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.45,
      .path = "/MNIST/N_45/In%d.bin",
      .label = "45% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.50,
      .path = "/MNIST/N_50/In%d.bin",
      .label = "50% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.55,
      .path = "/MNIST/N_55/In%d.bin",
      .label = "55% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.60,
      .path = "/MNIST/N_60/In%d.bin",
      .label = "60% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.65,
      .path = "/MNIST/N_65/In%d.bin",
      .label = "65% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.70,
      .path = "/MNIST/N_70/In%d.bin",
      .label = "70% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.75,
      .path = "/MNIST/N_75/In%d.bin",
      .label = "75% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.80,
      .path = "/MNIST/N_80/In%d.bin",
      .label = "80% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.85,
      .path = "/MNIST/N_85/In%d.bin",
      .label = "85% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.90,
      .path = "/MNIST/N_90/In%d.bin",
      .label = "90% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 0.95,
      .path = "/MNIST/N_95/In%d.bin",
      .label = "95% Noise",
      .statistics_array = {{0}}
    },
    { .noise = 1.00,
      .path = "/MNIST/N_100/In%d.bin",
      .label = "100% Noise",
      .statistics_array = {{0}}
    }
};

#else

static SbsStatistics statistics[] =
{
  { .input_pattern_first = 0,
    .input_pattern_last = 9999,
    .number_of_spikes = 1000
  }
};

static SbsBatchStatistics batch_statistics[] =
{
    { .noise = 0.50,
      .path = "/MNIST/N_50/In%d.bin",
      .label = "50% Noise",
      .statistics_array = {{0}}
    },
};

#endif //TAKE_ACCURACY_STATISTICS


void SbsStatistics_clean ()
{
  for (int i = 0; i < sizeof(statistics) / sizeof(SbsStatistics); i++)
  {
    statistics[i].result_correct_inferences = 0;
    statistics[i].result_total_inferences = 0;
    statistics[i].result_accuracy = 0;
  }
}

void SbsBatchStatistics_takeBatch (int batch)
{
  memcpy (&batch_statistics[batch].statistics_array, statistics, sizeof(statistics));
}

static void SnnApp_printStatistics ()
{
  int number_of_samples = sizeof(statistics) / sizeof(SbsStatistics);
  int number_of_batches = sizeof(batch_statistics) / sizeof(SbsBatchStatistics);

  printf ("#________SBS_ACCURACY_PLOT__________________________________________\n");

  printf ("import matplotlib.pyplot as pyplot\n\n\n");

  printf ("#________Accuracy_Vs_Spikes_________________________________________\n\n");

  for (int batch = 0; batch < number_of_batches; batch ++)
  {
    printf ("#%s\n", batch_statistics[batch].label);

    printf ("spikes_%d = [", batch);
    for (int loop = 0; loop < number_of_samples; loop++)
    {
      printf (" %d%c", batch_statistics[batch].statistics_array[loop].number_of_spikes,
              (loop + 1 < number_of_samples) ? ',' : ' ');
    }
    printf ("]\n");

    printf ("accuracy_%d = [", batch);
    for (int loop = 0; loop < number_of_samples; loop++)
    {
      printf (" %.2f%c", batch_statistics[batch].statistics_array[loop].result_accuracy,
              (loop + 1 < number_of_samples) ? ',' : ' ');
    }
    printf ("]\n");

    printf ("pyplot.plot(spikes_%d, accuracy_%d, label=\"%s\")\n", batch, batch, batch_statistics[batch].label);

    printf ("#______________________________________________\n\n");
  }

  printf ("pyplot.xlabel('Spikes')\n");
  printf ("pyplot.ylabel('Accuracy')\n");

  printf ("pyplot.title('Inference')\n");

  printf ("pyplot.legend()\n");
  printf ("pyplot.grid()\n");

  printf ("pyplot.show()\n\n\n\n");
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  printf ("#________Accuracy_Vs_Noise__________________________________________\n\n");

  for (int loop = 0; loop < number_of_samples; loop++)
  {
    printf ("#spikes = %d\n", batch_statistics[0].statistics_array[loop].number_of_spikes);

    printf ("noise_%d = [", loop);
    for (int batch = 0; batch < number_of_batches; batch ++)
    {
      printf (" %.2f%c", batch_statistics[batch].noise,
              (batch + 1 < number_of_batches) ? ',' : ' ');
    }
    printf ("]\n");

    printf ("accuracy_%d = [", loop);
    for (int batch = 0; batch < number_of_batches; batch ++)
    {
      printf (" %.2f%c", batch_statistics[batch].statistics_array[loop].result_accuracy,
              (batch + 1 < number_of_batches) ? ',' : ' ');
    }
    printf ("]\n");

    printf ("pyplot.plot(noise_%d, accuracy_%d, label=\"%d Spikes\")\n", loop, loop, batch_statistics[0].statistics_array[loop].number_of_spikes);

    printf ("#______________________________________________\n\n");
  }

  printf ("pyplot.xlabel('Noise')\n");
  printf ("pyplot.ylabel('Accuracy')\n");

  printf ("pyplot.title('Inference')\n");

  printf ("pyplot.legend()\n");
  printf ("pyplot.grid()\n");

  printf ("pyplot.show()\n\n");

  printf ("#________END__________________________________________\n\n");
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
}


Result SnnApp_run (void)
{
  int pattern_index;
  char string_text[128];
  float output_vector[10];
  uint16_t output_vector_size = 10;
  uint8_t input_label;
  uint8_t output_label;
  Timer * timer;
  Result result;

  // ********** Create SBS Neural Network **********
  printf ("\n==========  SbS Neural Network  ===============\n");
  printf ("\n==============  MNIST test  ===================\n");
  printf ("\n Building ...\n");
  /*_________________________________________________________________________*/

  SbsNetwork * network = sbs_new.Network ();

  /*_________________________________________________________________________*/

  SbsLayer * input_layer = sbs_new.InputLayer (HX_INPUT_LAYER,
                                               SBS_INPUT_LAYER_ROWS,
                                               SBS_INPUT_LAYER_COLUMNS,
                                               SBS_INPUT_LAYER_NEURONS);
  network->giveLayer (network, input_layer);

  /*_________________________________________________________________________*/

  SbsLayer * H1 = sbs_new.ConvolutionLayer (H1_CONVOLUTION_LAYER,
                                            SBS_H1_CONVOLUTION_LAYER_ROWS,
                                            SBS_H1_CONVOLUTION_LAYER_COLUMNS,
                                            SBS_H1_CONVOLUTION_LAYER_NEURONS,
                                            SBS_H1_CONVOLUTION_LAYER_KERNEL,
                                            ROW_SHIFT);
  H1->setEpsilon (H1, SBS_H1_CONVOLUTION_LAYER_EPSION);
  network->giveLayer (network, H1);

  SbsWeightMatrix P_IN_H1 = sbs_new.WeightMatrix (SBS_H1_CONVOLUTION_LAYER_KERNEL,
                                                  SBS_H1_CONVOLUTION_LAYER_KERNEL,
                                                  SBS_INPUT_LAYER_NEURONS,
                                                  SBS_H1_CONVOLUTION_LAYER_NEURONS,
                                                  4,
                                                  SBS_P_IN_H1_WEIGHTS_FILE);
  H1->giveWeights (H1, P_IN_H1);

  /*_________________________________________________________________________*/

  SbsLayer * H2 = sbs_new.PoolingLayer (H2_POOLING_LAYER,
                                        SBS_H2_POOLING_LAYER_ROWS,
                                        SBS_H2_POOLING_LAYER_COLUMNS,
                                        SBS_H2_POOLING_LAYER_NEURONS,
                                        SBS_H2_POOLING_LAYER_KERNEL,
                                        COLUMN_SHIFT);
  H2->setEpsilon (H2, SBS_H2_POOLING_LAYER_EPSION);
  network->giveLayer (network, H2);

  SbsWeightMatrix P_H1_H2 = sbs_new.WeightMatrix (SBS_H2_POOLING_LAYER_KERNEL,
                                                  SBS_H2_POOLING_LAYER_KERNEL,
                                                  SBS_H1_CONVOLUTION_LAYER_NEURONS,
                                                  SBS_H2_POOLING_LAYER_NEURONS,
                                                  4,
                                                  SBS_P_H1_H2_WEIGHTS_FILE);
  H2->giveWeights (H2, P_H1_H2);

  /*_________________________________________________________________________*/

  SbsLayer * H3 = sbs_new.ConvolutionLayer (H3_CONVOLUTION_LAYER,
                                            SBS_H3_CONVOLUTION_LAYER_ROWS,
                                            SBS_H3_CONVOLUTION_LAYER_COLUMNS,
                                            SBS_H3_CONVOLUTION_LAYER_NEURONS,
                                            SBS_H3_CONVOLUTION_LAYER_KERNEL,
                                            COLUMN_SHIFT);
  H3->setEpsilon (H3, SBS_H3_CONVOLUTION_LAYER_EPSION);
  network->giveLayer (network, H3);

  SbsWeightMatrix P_H2_H3 = sbs_new.WeightMatrix (SBS_H3_CONVOLUTION_LAYER_KERNEL,
                                                  SBS_H3_CONVOLUTION_LAYER_KERNEL,
                                                  SBS_H2_POOLING_LAYER_NEURONS,
                                                  SBS_H3_CONVOLUTION_LAYER_NEURONS,
                                                  4,
                                                  SBS_P_H2_H3_WEIGHTS_FILE);
  H3->giveWeights (H3, P_H2_H3);

  /*_________________________________________________________________________*/

  SbsLayer * H4 = sbs_new.PoolingLayer (H4_POOLING_LAYER,
                                        SBS_H4_POOLING_LAYER_ROWS,
                                        SBS_H4_POOLING_LAYER_COLUMNS,
                                        SBS_H4_POOLING_LAYER_NEURONS,
                                        SBS_H4_POOLING_LAYER_KERNEL,
                                        COLUMN_SHIFT);
  H4->setEpsilon (H4, SBS_H4_POOLING_LAYER_EPSION);
  network->giveLayer (network, H4);

  SbsWeightMatrix P_H3_H4 = sbs_new.WeightMatrix (SBS_H4_POOLING_LAYER_KERNEL,
                                                  SBS_H4_POOLING_LAYER_KERNEL,
                                                  SBS_H3_CONVOLUTION_LAYER_NEURONS,
                                                  SBS_H4_POOLING_LAYER_NEURONS,
                                                  4,
                                                  SBS_P_H3_H4_WEIGHTS_FILE);
  H4->giveWeights (H4, P_H3_H4);

  /*_________________________________________________________________________*/

  SbsLayer * H5 = sbs_new.FullyConnectedLayer (H5_FULLY_CONNECTED_LAYER,
                                               SBS_FULLY_CONNECTED_LAYER_NEURONS,
                                               SBS_H4_POOLING_LAYER_ROWS,
                                               ROW_SHIFT);
  H5->setEpsilon (H5, SBS_FULLY_CONNECTED_LAYER_EPSION);
  network->giveLayer (network, H5);

  SbsWeightMatrix P_H4_H5 = sbs_new.WeightMatrix (SBS_H4_POOLING_LAYER_ROWS,
                                                  SBS_H4_POOLING_LAYER_COLUMNS,
                                                  SBS_H4_POOLING_LAYER_NEURONS,
                                                  SBS_FULLY_CONNECTED_LAYER_NEURONS,
                                                  4,
                                                  SBS_P_H4_H5_WEIGHTS_FILE);
  H5->giveWeights (H5, P_H4_H5);


  /*_________________________________________________________________________*/

  SbsLayer * HY = sbs_new.OutputLayer (HY_OUTPUT_LAYER,
                                       SBS_OUTPUT_LAYER_NEURONS,
                                       ROW_SHIFT);
  HY->setEpsilon (HY, SBS_OUTPUT_LAYER_EPSION);
  network->giveLayer (network, HY);

  SbsWeightMatrix P_H5_HY = sbs_new.WeightMatrix (1,
                                                  1,
                                                  SBS_FULLY_CONNECTED_LAYER_NEURONS,
                                                  SBS_OUTPUT_LAYER_NEURONS,
                                                  4,
                                                  SBS_P_H5_HY_WEIGHTS_FILE);
  HY->giveWeights (HY, P_H5_HY);

  /*_________________________________________________________________________*/

  printf ("\n Inference ...\n");

  timer = Timer_new (1);

  Timer_start(timer);

  for (int batch = 0; batch < sizeof(batch_statistics) / sizeof(SbsBatchStatistics); batch++)
  {
    for (int loop = 0; loop < sizeof(statistics) / sizeof(SbsStatistics); loop++)
    {
      statistics[loop].result_accuracy = 0;
      statistics[loop].result_correct_inferences = 0;
      statistics[loop].result_total_inferences = 0;

      for (pattern_index = statistics[loop].input_pattern_first;
           pattern_index <= statistics[loop].input_pattern_last;
           pattern_index++)
      {
        sprintf (string_text,
                 batch_statistics[batch].path,
                 pattern_index);

        result = network->loadInput (network, string_text);

        if (result != OK)
        {
          printf ("\nError: loading pattern '%s'\n", string_text);
          continue;
        }

        printf ("\nSpikes: %d\n", statistics[loop].number_of_spikes);
        network->updateCycle (network, statistics[loop].number_of_spikes);

        statistics[loop].result_total_inferences ++;

        output_label = network->getInferredOutput (network);
        input_label = network->getInputLabel (network);

        if (output_label == input_label)
        {
          statistics[loop].result_correct_inferences ++;
          printf ("\nPASS\n");
        }
        else
        {
          printf ("\nMisclassification [label = %d]\n", input_label);
        }

        statistics[loop].result_accuracy =
            ((float) statistics[loop].result_correct_inferences)
                / ((float) statistics[loop].result_total_inferences);

        output_vector_size = sizeof(output_vector) / sizeof(float);
        network->getOutputVector (network, output_vector, output_vector_size);

        while (output_vector_size--)
        {
          float h = output_vector[output_vector_size];  //Ensure data alignment
          printf ("[%d] = %f\n", output_vector_size, h);
        }

        printf ("Accuracy: %.2f (%d/%d)\n",
                statistics[loop].result_accuracy,
                statistics[loop].result_correct_inferences,
                statistics[loop].result_total_inferences);

        printf ("Loop: %d, pattern: %d ( '%s' )\n", loop, pattern_index, string_text);

        //network->printStatistics (network);

        printf ("\n________________________________________________________________\n");
      }
    }
    SbsBatchStatistics_takeBatch (batch);
    SbsStatistics_clean ();
  }

  printf ("\nTime: %f Minutes\n", Timer_getCurrentTime(timer)/60);

  SnnApp_printStatistics ();

  printf ("\nEND\n");

  Timer_delete(&timer);

  network->delete (&network);

  return OK;
}

void SnnApp_dispose(void)
{
  SbsPlatform_shutdown ();
}

static SnnApp SnnApp_obj = { SnnApp_initialize,
                             SnnApp_run,
                             SnnApp_dispose };

SnnApp * SnnApp_instance(void)
{
  return & SnnApp_obj;
}
