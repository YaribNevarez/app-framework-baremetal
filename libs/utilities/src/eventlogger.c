
/************************ Event logger ***************************************/
#include "eventlogger.h"
#include "miscellaneous.h"

#include "string.h"
#include "stdlib.h"

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

inline void EventLogger_timeReset (void) __attribute__((always_inline));
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
  logger->point_array[logger->index].time = Timer_getCurrentTime(logger->timer);
  logger->point_array[logger->index].value = p;
  logger->index ++;
  if (logger->size <= logger->index)
    logger->index = 0;
}
