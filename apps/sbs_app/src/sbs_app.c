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

#include "xil_printf.h"
#include "xstatus.h"
#include "ff.h"


// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

// EUNUMERATIONS ---------------------------------------------------------------

// STRUCTS AND NAMESPACES ------------------------------------------------------

// DEFINITIONs -----------------------------------------------------------------

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
  SnnApp_initializeSD();
  return OK;
}

Result SnnApp_run (void)
{
  int pattern_index;
  char input_pattern_file_name[128];
  NeuronState * output_vector;
  uint16_t output_vector_size;
  uint8_t input_label;
  uint8_t output_label;
  Result rc;

  /*********************/
  // ********** Create SBS Neural Network **********
  printf ("\n==========  SbS Neural Network  ===============\n");
  printf ("\n==========  MNIST example  ====================\n");

  SbsStatistics_initialize ();

  rc = SbsHardware_initialize ();

  if (rc != OK)
  {
    printf ("Hardware error\n");
    return rc;
  }

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
                                                  SBS_P_H5_HY_WEIGHTS_FILE);
  HY->giveWeights (HY, P_H5_HY);

  /*_________________________________________________________________________*/

  HY->setLearningRule(HY, SBS_LEARNING_DELTA_MSE, 0.05, SBS_INPUT_PATTERN_LAST - SBS_INPUT_PATTERN_FIRST + 1);

  /*_________________________________________________________________________*/

  for (int loop = 0;; loop ++)
  {
    printf ("\n===============================================\n");
    printf ("\n Start loop: %d\n", loop);
    printf ("\n===============================================\n");
    for (pattern_index = SBS_INPUT_PATTERN_FIRST;
         pattern_index <= SBS_INPUT_PATTERN_LAST;
         pattern_index++)
    {
      printf ("\n===============================================\n");
      sprintf (input_pattern_file_name,
               SBS_INPUT_PATTERN_FORMAT_NAME,
               pattern_index);

      printf ("\nInput pattern: %s\n", input_pattern_file_name);
      network->loadInput (network, input_pattern_file_name);

      printf ("\n Loop: %d\n", loop);

      network->updateCycle (network, SBS_NETWORK_UPDATE_CYCLES);

      output_label = network->getInferredOutput (network);
      input_label = network->getInputLabel (network);

      if (output_label == input_label)
      {
    	  printf ("\n	PASS!");
      }
      else
      {
    	  printf ("\n	Misclassification!, %d != %d", input_label, output_label);
      }

      network->getOutputVector (network, &output_vector, &output_vector_size);

      while (output_vector_size--)
      {
        NeuronState h = output_vector[output_vector_size]; /* Ensure data alignment */
        printf ("\n [ %d ] = %f (0x%X)", output_vector_size, (float)h/(float)H_MAX, (unsigned int) h);
      }
      printf ("\n===============================================\n");
      network->printStatistics (network);
    }
    printf ("\n===============================================\n");
    printf ("\n End loop: %d\n", loop);
    printf ("\n===============================================\n");
  }
  network->delete (&network);

  SbsHardware_shutdown ();

  return rc;
}

void SnnApp_dispose(void)
{

}

static SnnApp SnnApp_obj = { SnnApp_initialize,
                             SnnApp_run,
                             SnnApp_dispose };

SnnApp * SnnApp_instance(void)
{
  return & SnnApp_obj;
}
