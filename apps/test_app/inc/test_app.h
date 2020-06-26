//------------------------------------------------------------------------------
/**
 *
 * @file: test_app.c
 *
 * @Created on: Jun 23rd, 2019
 * @Author: Yarib Nevarez
 *
 *
 * @brief - Memory bandwidth test application
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
#ifndef TEST_APP_H_
#define TEST_APP_H_

#ifdef __cplusplus
extern "C" {
#endif

// INCLUDES --------------------------------------------------------------------
//#include "xil_types.h"
#include "stdint.h"
#include "stddef.h"

#include "test_app_config.h"

#include "result.h"
// FORWARD DECLARATIONS --------------------------------------------------------

// TYPEDEFS AND DEFINES --------------------------------------------------------

// EUNUMERATIONS ---------------------------------------------------------------

// DECLARATIONS ----------------------------------------------------------------

typedef struct TestApp_ TestApp;

struct TestApp_
{
  Result  (* initialize) (TestApp * self);
  Result  (* run)        (TestApp * self);
  void    (* dispose)    (TestApp * self);
};

TestApp * TestApp_instance(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* TEST_APP_H_ */
