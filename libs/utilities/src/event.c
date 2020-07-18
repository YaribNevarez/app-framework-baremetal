/*
 * event.c
 *
 *  Created on: Feb 24th, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "event.h"
#include "miscellaneous.h"

#include "stdlib.h"
#include "string.h"
#include "stdio.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/*****************************************************************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

Event * Event_new (Event * parent, void * data)
{
  Event * event = malloc (sizeof(Event));
  ASSERT (event != NULL);

  if (event != NULL)
  {
    memset (event, 0, sizeof(Event));

    event->data = data;

    event->timer = Timer_new (1);

    ASSERT (event->timer != NULL);

    if (parent != NULL)
    {
      if (parent->first_child != NULL)
      {
        Event * child = parent->first_child;

        while (child->next != NULL)
        {
          child = child->next;
        }

        child->next = event;

        event->prev = child;
      }
      else
      {
        parent->first_child = event;
      }
      event->parent = parent;
    }
  }

  return event;
}

void Event_setParent (Event * event, Event * parent)
{
  ASSERT (event != NULL);
  ASSERT (event->parent == NULL)

  if ((event != NULL) && (event->parent == NULL))
  {
    event->parent = parent;

    if (parent->first_child != NULL)
    {
      Event * child = parent->first_child;

      while (child->next != NULL)
      {
        child = child->next;
      }

      child->next = event;

      event->prev = child;
    }
    else
    {
      parent->first_child = event;
    }
  }
}

void Event_delete (Event ** event)
{
  ASSERT (event != NULL);
  ASSERT (*event != NULL);

  if ((event != NULL) && (*event != NULL))
  {
    if (((*event)->parent != NULL) && ((*event)->parent->first_child == *event))
    {
      (*event)->parent->first_child = (*event)->next;

      if ((*event)->parent->first_child != NULL)
      {
        (*event)->parent->first_child->prev = NULL;
      }
    }

    if ((*event)->prev != NULL)
    {
      (*event)->prev->next = (*event)->next;
    }

    for (Event *child = (*event)->first_child; child != NULL; child = child->next)
    {
      child->parent = NULL;
    }

    Timer_delete (&(*event)->timer);

    free (*event);
    *event = NULL;
  }
}

void Event_start (Event * event)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    if (event->parent != NULL)
    {
      event->relative_offset = Event_getCurrentRelativeTime (event->parent);
      event->absolute_offset = event->parent->absolute_offset + event->relative_offset;
    }
    else
    {
      event->relative_offset = 0;
      event->absolute_offset = 0;
    }

    Timer_start (event->timer);

    event->latency = 0;
  }
}

double Event_getCurrentRelativeTime (Event * event)
{
  double current_time = 0;

  ASSERT (event != NULL);

  if (event != NULL)
  {
    current_time = Timer_getCurrentTime (event->timer);
  }

  return current_time;
}

double  Event_getCurrentAbsoluteTime (Event * event)
{
  double absolute_time = 0;

  ASSERT (event != NULL);

  if (event != NULL)
  {
    absolute_time = Event_getCurrentRelativeTime (event) + event->absolute_offset;
  }

  return absolute_time;
}

void Event_stop (Event * event)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    event->latency = Event_getCurrentRelativeTime (event);
  }
}

typedef enum
{
  NAV_CONTINUE,
  NAV_ABORT,
} NavigationReturn;

typedef NavigationReturn (*EventFunctionP) (Event *, void *);

static NavigationReturn Event_navegate (Event * event,
                                        EventFunctionP function,
                                        void * data)
{
  NavigationReturn result = NAV_ABORT;
  ASSERT (event != NULL);
  ASSERT (function != NULL);

  if ((event != NULL) && (function != NULL))
  {
    result = function (event, data);

    for (Event * child = event->first_child;
        (result != NAV_ABORT) && (child != NULL);
        child = child->next)
    {
      result = Event_navegate (child, function, data);
    }
  }

  return result;
}

typedef struct
{
  char text_offset[512];
  char text_latency[512];
  char text_name[512];
} EventScheduleData;

static NavigationReturn Event_schedulePrint (Event * event, void * data)
{
  NavigationReturn result = NAV_ABORT;
  ASSERT (event != NULL);
  ASSERT (data != NULL);

  if ((event != NULL) && (data != NULL))
  {
    EventScheduleData * scheduleData = (EventScheduleData*) data;
    char * text_offset = scheduleData->text_offset;
    char * text_latency = scheduleData->text_latency;
    char * text_name = scheduleData->text_name;

    sprintf (&text_offset[strlen (text_offset)], "%.3lf,", event->absolute_offset * 1000);
    sprintf (&text_latency[strlen (text_latency)], "%.3lf,", event->latency * 1000);
    sprintf (&text_name[strlen (text_name)], "\"%s\",", (char*) event->data);
    result = NAV_CONTINUE;
  }

  return result;
}

void Event_print (Event * event)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    EventScheduleData data;
    memset (&data, 0, sizeof(data));

    Event_navegate (event, Event_schedulePrint, &data);

    printf ("[%s]\n", data.text_offset);
    printf ("[%s]\n", data.text_latency);
    printf ("[%s]\n", data.text_name);
  }
}
