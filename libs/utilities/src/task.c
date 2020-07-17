/*
 * task.c
 *
 *  Created on: Feb 24th, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "task.h"
#include "miscellaneous.h"

#include "stdlib.h"
#include "string.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/*****************************************************************************/

/*typedef struct
 {
 Task *  parent;
 Task *  next;
 Task *  prev;
 Task *  first_child;

 Timer * timer;
 double  start_time;
 double  latency;
 } _Task;*/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

Task * Task_new (Task * parent)
{
  Task * task = malloc (sizeof(Task));
  ASSERT (task != NULL);

  if (task != NULL)
  {
    memset (task, 0, sizeof(Task));

    task->timer = Timer_new (1);

    ASSERT (task->timer != NULL);

    if (parent != NULL)
    {
      if (parent->first_child != NULL)
      {
        Task * child = parent->first_child;

        while (child->next != NULL)
        {
          child = child->next;
        }

        child->next = task;

        task->prev = child;
      }
      else
      {
        parent->first_child = task;
      }
      task->parent = parent;
    }
  }

  return task;
}

void Task_setParent (Task * task, Task * parent)
{
  ASSERT (task != NULL);
  ASSERT (task->parent == NULL)

  if ((task != NULL) && (task->parent == NULL))
  {
    task->parent = parent;

    if (parent->first_child != NULL)
    {
      Task * child = parent->first_child;

      while (child->next != NULL)
      {
        child = child->next;
      }

      child->next = task;

      task->prev = child;
    }
    else
    {
      parent->first_child = task;
    }
  }
}

void Task_delete (Task ** task)
{
  ASSERT (task != NULL);
  ASSERT (*task != NULL);

  if ((task != NULL) && (*task != NULL))
  {
    if (((*task)->parent != NULL) && ((*task)->parent->first_child == *task))
    {
      (*task)->parent->first_child = (*task)->next;

      if ((*task)->parent->first_child != NULL)
      {
        (*task)->parent->first_child->prev = NULL;
      }
    }

    if ((*task)->prev != NULL)
    {
      (*task)->prev->next = (*task)->next;
    }

    for (Task *child = (*task)->first_child; child != NULL; child = child->next)
    {
      child->parent = NULL;
    }

    Timer_delete (&(*task)->timer);

    free (*task);
    *task = NULL;
  }
}

void Task_start (Task * task)
{
  ASSERT (task != NULL);

  if (task != NULL)
  {
    if (task->parent != NULL)
    {
      task->start_time = Task_getCurrentTime (task->parent);
    }
    else
    {
      task->start_time = 0;
    }

    Timer_start (task->timer);

    task->latency = 0;
  }
}

double Task_getCurrentTime (Task * task)
{
  double current_time = 0;

  ASSERT (task != NULL);

  if (task != NULL)
  {
    current_time = Timer_getCurrentTime (task->timer);
  }
  return current_time;
}

void Task_stop (Task * task)
{
  ASSERT (task != NULL);

  if (task != NULL)
  {
    task->latency = Task_getCurrentTime (task);
  }
}

