//------------------------------------------------------------------------------
/**
 *
 * @file: test_platform.h
 *
 * @Created on: Jun 23rd, 2019
 * @Author: Yarib Nevarez
 *
 *
 * @brief - Memory bandwidth test application platform
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
#ifndef TEST_PLATFORM_H_
#define TEST_PLATFORM_H_

#include "xparameters.h"

/////// DMA hardware parameters ------------------------------------------------
#define DMA_DEVICE_ID       XPAR_AXI_DMA_0_DEVICE_ID
#define DMA_TX_INTR_ID      XPAR_FABRIC_AXI_DMA_0_MM2S_INTROUT_INTR
#define DMA_RX_INTR_ID      XPAR_FABRIC_AXI_DMA_0_S2MM_INTROUT_INTR

/////// Kernel hardware parameters ---------------------------------------------
#define KERNEL_DEVICE_ID    XPAR_TEST_MODULE_0_DEVICE_ID
#define KERNEL_INTR_ID      XPAR_FABRIC_TEST_MODULE_0_INTERRUPT_INTR

#endif /* TEST_PLATFORM_H_ */
