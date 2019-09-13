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

// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

// EUNUMERATIONS ---------------------------------------------------------------

// STRUCTS AND NAMESPACES ------------------------------------------------------

// DEFINITIONs -----------------------------------------------------------------

Result SnnApp_initialize(void)
{
  return OK;
}

Result SnnApp_run(void)
{
  //sbs_test();
  NeuronState * output_vector;
  uint16_t output_vector_size;

  sgenrand(666);

  /*********************/
  // ********** Create SBS Neural Network **********
  printf("\n==========  SbS Neural Network  ===============\n");
  printf("\n==========  MNIST example  ====================\n");

  SbsNetwork * network = SbsNetwork_vtable.new();

  // Instantiate SBS Network objects
  SbsInputLayer input_layer = SbsInputLayer_new(24, 24, 50);
  network->giveLayer(network, input_layer);

  SbsWeightMatrix P_IN_H1 = SbsWeightMatrix_new(2 * 5 * 5, 32, "/home/nevarez/Downloads/MNIST/W_X_H1_Iter0.bin");

  SbsConvolutionLayer H1 = SbsConvolutionLayer_new(24, 24, 32, 1, ROW_SHIFT, 50);
  SbsBaseLayer_setEpsilon(H1, 0.1);
  SbsBaseLayer_giveWeights(H1, P_IN_H1);
  network->giveLayer(network, H1);

  SbsWeightMatrix P_H1_H2 = SbsWeightMatrix_new(32 * 2 * 2, 32, "/home/nevarez/Downloads/MNIST/W_H1_H2.bin");

  SbsPoolingLayer H2 = SbsPoolingLayer_new(12, 12, 32, 2, COLUMN_SHIFT, 32);
  SbsBaseLayer_setEpsilon(H2, 0.1 / 4.0);
  SbsBaseLayer_giveWeights(H2, P_H1_H2);
  network->giveLayer(network, H2);

  SbsWeightMatrix P_H2_H3 = SbsWeightMatrix_new(32 * 5 * 5, 64, "/home/nevarez/Downloads/MNIST/W_H2_H3_Iter0.bin");

  SbsConvolutionLayer H3 = SbsConvolutionLayer_new(8, 8, 64, 5, COLUMN_SHIFT, 32);
  SbsBaseLayer_setEpsilon(H3, 0.1 / 25.0);
  SbsBaseLayer_giveWeights(H3, P_H2_H3);
  network->giveLayer(network, H3);

  SbsWeightMatrix P_H3_H4 = SbsWeightMatrix_new(64 * 2 * 2, 64, "/home/nevarez/Downloads/MNIST/W_H3_H4.bin");

  SbsPoolingLayer H4 = SbsPoolingLayer_new(4, 4, 64, 2, COLUMN_SHIFT, 64);
  SbsBaseLayer_setEpsilon(H4, 0.1 / 4.0);
  SbsBaseLayer_giveWeights(H4, P_H3_H4);
  network->giveLayer(network, H4);

  SbsWeightMatrix P_H4_H5 = SbsWeightMatrix_new(64 * 4 * 4, 1024, "/home/nevarez/Downloads/MNIST/W_H4_H5_Iter0.bin");

  SbsFullyConnectedLayer H5 = SbsFullyConnectedLayer_new(1024, 4, ROW_SHIFT, 64);
  SbsBaseLayer_setEpsilon(H5, 0.1 / 16.0);
  SbsBaseLayer_giveWeights(H5, P_H4_H5);
  network->giveLayer(network, H5);

  SbsWeightMatrix P_H5_HY = SbsWeightMatrix_new(1024, 10, "/home/nevarez/Downloads/MNIST/W_H5_HY_Iter0.bin");

  SbsOutputLayer HY = SbsOutputLayer_new(10, ROW_SHIFT, 0);
  SbsBaseLayer_setEpsilon(HY, 0.1);
  SbsBaseLayer_giveWeights(HY, P_H5_HY);
  network->giveLayer(network, HY);

    // Perform Network load pattern and update cycle
  network->loadInput(network, "/home/nevarez/Downloads/MNIST/Pattern/Input_1.bin");
  network->updateCycle(network, 1000);

  printf("\n==========  Results ===========================\n");

  printf("\n Output value: %d \n", network->getInferredOutput(network));
  printf("\n Label value: %d \n", network->getInputLabel(network));

  network->getOutputVector(network, &output_vector, &output_vector_size);

  printf("\n==========  Output layer values ===============\n");

  while (output_vector_size --)
  {
    printf(" [ %d ] = %.6f\n", output_vector_size, output_vector[output_vector_size]);
  }

  SbsNetwork_vtable.delete(&network);
  /*********************/
  return OK;
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
