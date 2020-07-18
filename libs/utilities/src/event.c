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
      event->offset = Event_getCurrentTime (event->parent);
    }
    else
    {
      event->offset = 0;
    }

    Timer_start (event->timer);

    event->latency = 0;
  }
}

double Event_getCurrentTime (Event * event)
{
  double current_time = 0;

  ASSERT (event != NULL);

  if (event != NULL)
  {
    current_time = Timer_getCurrentTime (event->timer);
  }
  return current_time;
}

void Event_stop (Event * event)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    event->latency = Event_getCurrentTime (event);
  }
}

static void Event_navegatePrint (Event * event,
                                 double carry_offset,
                                 char * text_offset,
                                 char * text_latency,
                                 char * text_name)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    carry_offset += event->offset;

    sprintf (&text_offset[strlen (text_offset)], "%.8lf,", carry_offset);
    sprintf (&text_latency[strlen (text_latency)], "%.8lf,", event->latency);
    sprintf (&text_name[strlen (text_name)], "\"%s\",", (char*) event->data);

    for (Event * child = event->first_child; child != NULL; child = child->next)
    {
      Event_navegatePrint (child,
                           carry_offset,
                           text_offset,
                           text_latency,
                           text_name);
    }
  }
}

void Event_print (Event * event)
{
  ASSERT (event != NULL);

  if (event != NULL)
  {
    static char text_offset[512];
    static char text_latency[512];
    static char text_name[512];

    memset (text_offset, 0, sizeof(text_offset));
    memset (text_latency, 0, sizeof(text_latency));
    memset (text_name, 0, sizeof(text_name));

    Event_navegatePrint (event, 0, text_offset, text_latency, text_name);

    printf ("[%s]\n", text_offset);
    printf ("[%s]\n", text_latency);
    printf ("[%s]\n", text_name);
  }
}
