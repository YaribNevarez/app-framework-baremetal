//------------------------------------------------------------------------------
/**
 *
 * @file: sbs_platform.h
 *
 * @Created on: March 7th, 2019
 * @Author: Yarib Nevarez
 *
 *
 * @brief - Spike by Spike Neural Network platfrom
 * <Requirement Doc Reference>
 * <Design Doc Reference>
 *
 */
//------------------------------------------------------------------------------
// IFNDEF ----------------------------------------------------------------------
#ifndef SBS_PLATFORM_H_
#define SBS_PLATFORM_H_


#define ACCELERATOR_0     HX_INPUT_LAYER
//#define ACCELERATOR_1     H1_CONVOLUTION_LAYER | H2_POOLING_LAYER | H3_CONVOLUTION_LAYER | H4_POOLING_LAYER | H5_FULLY_CONNECTED_LAYER | HY_OUTPUT_LAYER
#define ACCELERATOR_M     H1_CONVOLUTION_LAYER | H2_POOLING_LAYER | H3_CONVOLUTION_LAYER | H4_POOLING_LAYER | H5_FULLY_CONNECTED_LAYER | HY_OUTPUT_LAYER

/*___________________________________________________________________________*/

#define MT19937_SEED      (666)


#endif /* SBS_PLATFORM_H_ */
