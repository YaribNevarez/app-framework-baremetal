/*
 * task.h
 *
 *  Created on: Feb 24th, 2020
 *      Author: Yarib Nevarez
 */
#ifndef LIBS_UTILITIES_TASK_H_
#define LIBS_UTILITIES_TASK_H_

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/

#include "timer.h"
#include "xil_types.h"

/***************** Macros (Inline Functions) Definitions *********************/
typedef struct _Task Task;
/**************************** Type Definitions *******************************/
struct  _Task
{
  Task *  parent;
  Task *  next;
  Task *  prev;
  Task *  first_child;

  Timer * timer;
  double  start_time;
  double  latency;
};

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

Task *  Task_new             (Task * parent);
void    Task_delete          (Task ** task);
void    Task_setParent       (Task * task, Task * parent);
void    Task_start           (Task * task);
void    Task_stop            (Task * task);
double  Task_getCurrentTime  (Task * task);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LIBS_UTILITIES_TIMER_H_ */
