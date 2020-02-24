/*
 * ToolCom.c
 *
 *  Created on: 3 de feb. de 2017
 *      Author: Yarib Nevarez
 */

#include "toolcom.h"
#include "serialport.h"
#include "sleep.h"
#include "string.h"

static void     ToolCom_clearTrace(ToolTrace trace);
static uint8_t  ToolCom_plotSamples(ToolTrace trace, double * array, uint32_t length);
static void     ToolCom_setVisible(ToolTrace trace, uint8_t visible);
static void     ToolCom_setTime(ToolTrace trace, double time);
static void     ToolCom_setStepTime(ToolTrace trace, double time);
static void     ToolCom_textMsg(uint8_t id, char * msg);
static void     ToolCom_setProgressCallback(ToolProgressCallback function, void * data);

static ToolProgressCallback progressCallback = NULL;
static void * progressCallbackData = NULL;

static uint8_t ToolCom_doubleBulkLength = 16;

#define CMD_CLEAR         0x00
#define CMD_PLOT          0x01
#define CMD_SET_VISIBLE   0x02
#define CMD_SET_STEP_TIME 0x03
#define CMD_SET_TIME      0x04
#define CMD_TEXT_MSG      0x05

static ToolCom ToolCom_obj =
{
    ToolCom_clearTrace,
    ToolCom_plotSamples,
    ToolCom_setVisible,
    ToolCom_setTime,
    ToolCom_setStepTime,
    ToolCom_textMsg,
    ToolCom_setProgressCallback
};

ToolCom * ToolCom_instance(void)
{
  return &ToolCom_obj;
}

static void ToolCom_clearTrace(ToolTrace trace)
{
  uint8_t cmd[] = {CMD_CLEAR, 0};
  cmd[1] = trace;
  SerialPort_instance ()->sendFrameCommand (cmd, sizeof(cmd), NULL, 0);
}

static uint8_t ToolCom_plotSamples(ToolTrace trace,
                                   double * array,
                                   uint32_t length)
{
  uint8_t rc;
  uint8_t cmd[] = {CMD_PLOT, 0, 0};
  uint32_t i, len;

  cmd[1] = trace;

  rc = array != NULL;

  if (rc)
  {
    for (i = 0; i < length && rc; i += len)
    {
      if (i + ToolCom_doubleBulkLength < length)
      {
        len = ToolCom_doubleBulkLength;
      }
      else
      {
        len = length - i;
      }
      cmd[2] = len;
      SerialPort_instance ()->sendFrameCommand (cmd, sizeof(cmd),
                                                (uint8_t *) &array[i],
                                                sizeof(double) * len);

      if (progressCallback != NULL)
      {
        rc = progressCallback (progressCallbackData, i + len, length);
      }
      usleep (50000);
    }
  }

  return rc;
}

static void ToolCom_setVisible(ToolTrace trace, uint8_t visible)
{
  uint8_t cmd[] = {CMD_SET_VISIBLE, 0, 0};
  cmd[1] = trace;
  cmd[2] = visible;
  SerialPort_instance ()->sendFrameCommand (cmd, sizeof(cmd), NULL, 0);
}

static void ToolCom_setTime(ToolTrace trace, double time)
{
  uint8_t cmd[] = {CMD_SET_TIME, 0};
  cmd[1] = trace;
  SerialPort_instance ()->sendFrameCommand (cmd,
                                            sizeof(cmd),
                                            (uint8_t *) &time,
                                            sizeof(double));
}

static void ToolCom_setStepTime(ToolTrace trace, double time)
{
  uint8_t cmd[] = {CMD_SET_STEP_TIME, 0};
  cmd[1] = trace;
  SerialPort_instance ()->sendFrameCommand (cmd,
                                            sizeof(cmd),
                                            (uint8_t *) &time,
                                            sizeof(double));
}

static void ToolCom_textMsg(uint8_t id, char * msg)
{
  uint8_t cmd[] = {CMD_TEXT_MSG, 0, 0};
  cmd[1] = id;
  cmd[2] = strlen (msg);
  SerialPort_instance ()->sendFrameCommand (cmd,
                                            sizeof(cmd),
                                            (uint8_t *) msg,
                                            sizeof(char) * strlen (msg));
}

static void ToolCom_setProgressCallback(ToolProgressCallback function,
                                        void * data)
{
  progressCallback = function;
  progressCallbackData = data;
}
