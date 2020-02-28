
/************************ Event logger ***************************************/
#include "eventlogger.h"
#include "miscellaneous.h"
#include "toolcom.h"

#include "string.h"
#include "stdlib.h"

#define MUTEX_LOCK(mutex)       while (mutex); (mutex) = 1;
#define MUTEX_UNLOCK(mutex)     (mutex) = 0;

Timer * EventLogger_timer = NULL;

EventLogger * EventLogger_new (int num_logs)
{
  EventLogger * logger = NULL;

  if (EventLogger_timer == NULL)
    EventLogger_timer = Timer_new (1);

  ASSERT (EventLogger_timer != NULL);

  logger = malloc (sizeof(EventLogger) + (num_logs - 1) * sizeof(Point));

  ASSERT (logger != NULL);

  memset (logger, 0, sizeof(EventLogger) + (num_logs - 1) * sizeof(Point));

  logger->size = num_logs;
  logger->timer = EventLogger_timer;

  return logger;
}

void EventLogger_delete (EventLogger ** logger)
{
  ASSERT (logger != NULL);
  ASSERT (*logger != NULL);

  if ((logger != NULL) && (*logger != NULL))
  {
    free (*logger);
    *logger = NULL;
  }
}

inline void EventLogger_timeReset (void)
{
  if (EventLogger_timer == NULL)
    EventLogger_timer = Timer_new(1);

  Timer_start(EventLogger_timer);
}

inline void EventLogger_logPoint(EventLogger * logger, double p) __attribute__((always_inline));
inline void EventLogger_logPoint(EventLogger * logger, double p)
{
  ASSERT(logger != NULL);
  if (logger != NULL)
  {
    MUTEX_LOCK(logger->mutex);

    logger->point_array[logger->index].time = Timer_getCurrentTime(logger->timer);
    logger->point_array[logger->index].value = p;
    logger->index ++;
    if (logger->size <= logger->index)
      logger->index = 0;

    MUTEX_UNLOCK(logger->mutex);
  }
}

inline void EventLogger_logTransition(EventLogger * logger, EventTransition event) __attribute__((always_inline));
inline void EventLogger_logTransition(EventLogger * logger, EventTransition event)
{
  ASSERT(logger != NULL);
  if (logger != NULL)
  {
    MUTEX_LOCK(logger->mutex);
    if (logger->index + 2 <= logger->size)
    {
      double time = Timer_getCurrentTime (logger->timer);

      logger->point_array[logger->index].time = time;
      logger->point_array[logger->index].value = event == RISE_EVENT? 0.0 : 1.0;
      logger->index++;

      logger->point_array[logger->index].time = time;
      logger->point_array[logger->index].value = event == RISE_EVENT? 1.0 : 0.0;
      logger->index++;

      if (logger->size <= logger->index) logger->index = 0;
    }
    MUTEX_UNLOCK(logger->mutex);
  }
}

void EventLogger_flush(EventLogger * logger)
{
  ASSERT(logger != NULL);
  if (logger != NULL)
  {
    MUTEX_LOCK(logger->mutex);

    ToolCom_instance ()->plotSamples (0,
                                      (double *) logger->point_array,
                                      logger->index * (sizeof(Point) / sizeof(double)));
    logger->index = 0;

    MUTEX_UNLOCK(logger->mutex);
  }
}
