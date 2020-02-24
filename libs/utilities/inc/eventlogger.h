/*
 * eventlogger.h
 *
 *  Created on: Feb 24th, 2020
 *      Author: Yarib Nevarez
 */
#ifndef LIBS_UTILITIES_EVENTLOGGER_H_
#define LIBS_UTILITIES_EVENTLOGGER_H_

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#include "timer.h"
/***************** Macros (Inline Functions) Definitions *********************/


/**************************** Type Definitions *******************************/
typedef struct
{
  double value;
  double time;
} Point;

typedef struct
{
  Timer * timer;
  int index;
  int size;
  int mutex;
  Point point_array[1];
} EventLogger;

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LIBS_UTILITIES_EVENTLOGGER_H_ */
