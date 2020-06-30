//------------------------------------------------------------------------------
/**
 *
 * @file: test_app_config.c
 *
 * @Created on: Jun 23rd, 2019
 * @Author: Yarib Nevarez
 *
 *
 * @brief - Memory bandwidth test application configuration
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
#ifndef TEST_APP_CONFIG_H_
#define TEST_APP_CONFIG_H_

// INCLUDES --------------------------------------------------------------------

// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

/////// Kernel hardware parameters ---------------------------------------------
#define DATA_SIZE           (32/8)
#define BUFFER_LENGTH       4
#define MAX_BUFFER_SIZE     (256 * 1024)
#define BUFFER_IN_ADDRESS   (void *) (XPAR_PS7_DDR_0_S_AXI_HIGHADDR - 1024*1024 + 1)
#define BUFFER_OUT_ADDRESS  (void *) (XPAR_PS7_DDR_0_S_AXI_HIGHADDR -  512*1024 + 1)

// -----------------------------------------------------------------------------
//  Test cases (ordered list)
//
//  MASTER_DIRECT
//  MASTER_DIRECT_PIPELINE
//  MASTER_CACHED
//  MASTER_CACHED_PIPELINED
//  MASTER_CACHED_BURST
//  MASTER_SEND_BURST
//  MASTER_RETRIEVE_BURST
//  MASTER_SEND
//  MASTER_RETRIEVE
//  MASTER_SEND_PIPELINED
//  MASTER_RETRIEVE_PIPELINED
//  STREAM_DIRECT
//  STREAM_DIRECT_PIPELINED
//  STREAM_CACHED
//  STREAM_CACHED_PIPELINED
//  STREAM_SEND
//  STREAM_RETRIEVE
//  STREAM_SEND_PIPELINED
//  STREAM_RETRIEVE_PIPELINED
//
//
#define FIRST_TEST_CASE   MASTER_DIRECT
#define LAST_TEST_CASE    STREAM_RETRIEVE_PIPELINED

#endif /* TEST_APP_CONFIG_H_ */
