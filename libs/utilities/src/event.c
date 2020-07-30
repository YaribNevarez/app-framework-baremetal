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

/*****************************************************************************/

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

/*****************************************************************************/

typedef char TextLines[4][512];

static NavigationReturn Event_collectScheduleData (Event * event, void * data)
{
  NavigationReturn result = NAV_ABORT;
  ASSERT (event != NULL);
  ASSERT (data != NULL);

  if ((event != NULL) && (data != NULL))
  {
    if (((event->first_child == NULL) && (0.0 < event->latency))
        || (event->first_child->first_child == NULL)
        || (event->parent == NULL))
    {
      TextLines * text = (TextLines*) data;
      char * layer_name;
      char * color;

      if (event->first_child == NULL)
      { // Hardware
        layer_name = event->parent->parent->data;
        color = "#1864ab";
      }
      else if (event->first_child->first_child == NULL)
      { // Software
        layer_name = event->parent->data;
        color = "#4a98c9";
      }
      else
      { // Network
        layer_name = "";
        color = "#94c4df";
      }

      sprintf (&(*text)[0][strlen ((*text)[0])], "%.3lf, ", event->absolute_offset * 1000);
      sprintf (&(*text)[1][strlen ((*text)[1])], "%.3lf, ", event->latency * 1000);
      sprintf (&(*text)[2][strlen ((*text)[2])], "\"%s_%s\", ", layer_name, (char*) event->data);
      sprintf (&(*text)[3][strlen ((*text)[3])], "\"%s\", ", color);
    }
    result = NAV_CONTINUE;
  }

  return result;
}

static NavigationReturn Event_collectLatencyData (Event * event, void * data)
{
  NavigationReturn result = NAV_ABORT;
  ASSERT (event != NULL);
  ASSERT (data != NULL);

  if ((event != NULL) && (data != NULL))
  {
    if (   (event->first_child == NULL)
        && (event->parent != NULL)
        && (event->parent->parent != NULL))
    { /* Then is a hardware event */
      TextLines * text = (TextLines*) data;

      sprintf (&(*text)[0][strlen ((*text)[0])], "%.3lf, ", event->parent->absolute_offset * 1000);
      sprintf (&(*text)[1][strlen ((*text)[1])], "%.3lf, ", event->parent->latency * 1000);
      sprintf (&(*text)[2][strlen ((*text)[2])], "%.3lf, ", event->latency * 1000);
      sprintf (&(*text)[3][strlen ((*text)[3])], "\"%s\", ", (char*) event->parent->parent->data);
    }
    result = NAV_CONTINUE;
  }

  return result;
}

void Event_print (Event * event)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    TextLines data;

    memset (data, 0, sizeof(data));
    Event_navegate (event, Event_collectScheduleData, data);

    printf ("\nSchedule\n");
    printf ("Absolute offset: [%s]\n", data[0]);
    printf ("Latency:         [%s]\n", data[1]);
    printf ("Name:            [%s]\n", data[2]);
    printf ("Color:           [%s]\n", data[3]);

    memset (data, 0, sizeof(data));
    Event_navegate (event, Event_collectLatencyData, data);

    printf ("\nPerformance\n");
    printf ("II offset:   [%s]\n", data[0]);
    printf ("SW Latency:  [%s]\n", data[1]);
    printf ("HW Latency:  [%s]\n", data[2]);
    printf ("Name:        [%s]\n", data[3]);
  }
}
