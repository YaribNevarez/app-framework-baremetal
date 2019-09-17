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
 * @copyright Copyright [2019] Institute for Theoretical Electrical Engineering
 *                             and Microelectronics (ITEM)
 * All Rights Reserved.
 *
 *
 */
//------------------------------------------------------------------------------
// INCLUDES --------------------------------------------------------------------
#include "sbs_neural_network.h"
#include "sbs_app.h"
#include "stdio.h"

#ifdef USE_XILINX

#include "platform.h"
#include "xil_printf.h"
#include "IO.h"
#include "ff.h"

#endif

// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

// EUNUMERATIONS ---------------------------------------------------------------

// STRUCTS AND NAMESPACES ------------------------------------------------------

// DEFINITIONs -----------------------------------------------------------------
#ifdef USE_XILINX
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
#endif

Result SnnApp_initialize(void)
{
#ifdef USE_XILINX
    init_platform();
    SnnApp_initializeSD();
#endif

  return OK;
}

Result SnnApp_run(void)
{
  NeuronState * output_vector;
  uint16_t output_vector_size;

  /*********************/
  // ********** Create SBS Neural Network **********
  printf("\n==========  SbS Neural Network  ===============\n");
  printf("\n==========  MNIST example  ====================\n");

  SbsNetwork * network = sbs_new.Network();

  // Instantiate SBS Network objects

  /** Layer = 24x24x50, Spike = 24x24, Weight = 0 **/
  SbsLayer * input_layer = sbs_new.InputLayer(24, 24, 50);
  network->giveLayer(network, input_layer);

  SbsWeightMatrix P_IN_H1 = sbs_new.WeightMatrix(2 * 5 * 5, 32, SBS_P_IN_H1_WEIGHTS_FILE);

  /** Layer = 24x24x32, Spike = 24x24, Weight = 50x32 **/
  SbsLayer * H1 = sbs_new.ConvolutionLayer(24, 24, 32, 1, ROW_SHIFT, 50);
  H1->setEpsilon(H1, 0.1);
  H1->giveWeights(H1, P_IN_H1);
  network->giveLayer(network, H1);

  SbsWeightMatrix P_H1_H2 = sbs_new.WeightMatrix(32 * 2 * 2, 32, SBS_P_H1_H2_WEIGHTS_FILE);

  /** Layer = 12x12x32, Spike = 12x12, Weight = 128x32 **/
  SbsLayer * H2 = sbs_new.PoolingLayer(12, 12, 32, 2, COLUMN_SHIFT, 32);
  H2->setEpsilon(H2, 0.1 / 4.0);
  H2->giveWeights(H2, P_H1_H2);
  network->giveLayer(network, H2);

  SbsWeightMatrix P_H2_H3 = sbs_new.WeightMatrix(32 * 5 * 5, 64, SBS_P_H2_H3_WEIGHTS_FILE);

  /** Layer = 8x8x64, Spike = 8x8, Weight = 800x64 **/
  SbsLayer * H3 = sbs_new.ConvolutionLayer(8, 8, 64, 5, COLUMN_SHIFT, 32);
  H3->setEpsilon(H3, 0.1 / 25.0);
  H3->giveWeights(H3, P_H2_H3);
  network->giveLayer(network, H3);

  SbsWeightMatrix P_H3_H4 = sbs_new.WeightMatrix(64 * 2 * 2, 64, SBS_P_H3_H4_WEIGHTS_FILE);

  /** Layer = 4x4x64, Spike = 4x4, Weight = 256x64 **/
  SbsLayer * H4 = sbs_new.PoolingLayer(4, 4, 64, 2, COLUMN_SHIFT, 64);
  H4->setEpsilon(H4, 0.1 / 4.0);
  H4->giveWeights(H4, P_H3_H4);
  network->giveLayer(network, H4);

  SbsWeightMatrix P_H4_H5 = sbs_new.WeightMatrix(64 * 4 * 4, 1024, SBS_P_H4_H5_WEIGHTS_FILE);

  /** Layer = 1x1x1024, Spike = 1x1, Weight = 1024x1024 **/
  SbsLayer * H5 = sbs_new.FullyConnectedLayer(1024, 4, ROW_SHIFT, 64);
  H5->setEpsilon(H5, 0.1 / 16.0);
  H5->giveWeights(H5, P_H4_H5);
  network->giveLayer(network, H5);

  SbsWeightMatrix P_H5_HY = sbs_new.WeightMatrix(1024, 10, SBS_P_H5_HY_WEIGHTS_FILE);

  /** Layer = 1x1x10, Spike = 1x1, Weight = 1024x10 **/
  SbsLayer * HY = sbs_new.OutputLayer(10, ROW_SHIFT, 0);
  HY->setEpsilon(HY, 0.1);
  HY->giveWeights(HY, P_H5_HY);
  network->giveLayer(network, HY);

    // Perform Network load pattern and update cycle
  network->loadInput(network, SBS_INPUT_PATTERN_FILE);
  network->updateCycle(network, 1000);

  printf("\n==========  Results ===========================\n");

  printf("\n Output value: %d \n", network->getInferredOutput(network));
  printf("\n Label value: %d \n", network->getInputLabel(network));

  network->getOutputVector(network, &output_vector, &output_vector_size);

  printf("\n==========  Output layer values ===============\n");

  while (output_vector_size --)
  {
    NeuronState h = output_vector[output_vector_size]; /* Ensure data alignment */
    printf(" [ %d ] = %.6f\n", output_vector_size, h);
  }

  printf("\n===============================================\n");

#ifdef USE_XILINX
  printf("\n Pool size: %d \n", network->getMemorySize(network));
#else
  printf("\n Pool size: %ld \n", network->getMemorySize(network));
#endif

  network->delete(&network);

  return OK;
}

void SnnApp_dispose(void)
{
#ifdef USE_XILINX
	cleanup_platform();
#endif
}

static SnnApp SnnApp_obj = { SnnApp_initialize,
                             SnnApp_run,
                             SnnApp_dispose };

SnnApp * SnnApp_instance(void)
{
  return & SnnApp_obj;
}
