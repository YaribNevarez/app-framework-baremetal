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
#define ACCELERATOR_1     H1_CONVOLUTION_LAYER | HY_OUTPUT_LAYER | H5_FULLY_CONNECTED_LAYER
#define ACCELERATOR_2     H2_POOLING_LAYER | H4_POOLING_LAYER
#define ACCELERATOR_3     H3_CONVOLUTION_LAYER
//#define ACCELERATOR_4     H3_CONVOLUTION_LAYER
//#define ACCELERATOR_5     H5_FULLY_CONNECTED_LAYER

/*___________________________________________________________________________*/

#define MT19937_SEED      (666)


#endif /* SBS_PLATFORM_H_ */
