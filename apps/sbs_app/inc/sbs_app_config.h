//------------------------------------------------------------------------------
/**
 *
 * @file: sbs_app_config.h
 *
 * @Created on: September 15th, 2019
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
 */
//------------------------------------------------------------------------------
// IFNDEF ----------------------------------------------------------------------
#ifndef SBS_APP_CONFIG_H_
#define SBS_APP_CONFIG_H_

// INCLUDES --------------------------------------------------------------------

// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------
//#define USE_XILINX

#define SBS_INPUT_PATTERN_FILE   "/MNIST/Pattern/Input_1.bin"

#define SBS_INPUT_PATTERN_FORMAT_NAME "/MNIST/Pattern/Input_%d.bin"
#define SBS_INPUT_PATTERN_FIRST       1
#define SBS_INPUT_PATTERN_LAST        50

#define SBS_P_IN_H1_WEIGHTS_FILE "/MNIST/W_X_H1.bin"
#define SBS_P_H1_H2_WEIGHTS_FILE "/MNIST/W_H1_H2.bin"
#define SBS_P_H2_H3_WEIGHTS_FILE "/MNIST/W_H2_H3.bin"
#define SBS_P_H3_H4_WEIGHTS_FILE "/MNIST/W_H3_H4.bin"
#define SBS_P_H4_H5_WEIGHTS_FILE "/MNIST/W_H4_H5.bin"
#define SBS_P_H5_HY_WEIGHTS_FILE "/MNIST/W_H5_HY.bin"

// EUNUMERATIONS ---------------------------------------------------------------

// DECLARATIONS ----------------------------------------------------------------

#endif /* SBS_APP_CONFIG_H_ */
