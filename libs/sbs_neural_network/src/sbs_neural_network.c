/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: yarib
 */


#include "sbs_neural_network.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "stddef.h"
#include "stdarg.h"

#define ASSERT(expr)  assert(expr)

typedef struct
{
  float * data;
  uint8_t dimensional;
  uint8_t dimensional_size[1]; /*[0] = x, [1] = y, [2] = z... [n] = N*/
} Multivector;

typedef enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
} WeightShift;

typedef struct
{
  Multivector * state_matrix;
  Multivector * weight_matrix;
  Multivector * spike_matrix;

  uint16_t    kernel_size;
  uint16_t    kernel_stride;
  uint16_t    neurons_prev_Layer;
  WeightShift weight_shift;
  float       epsilon;
} SbsBaseLayer;

/*****************************************************************************/

void * ram_block_request(size_t size)
{
  return malloc(size);
}

/*****************************************************************************/

Multivector * Multivector_new(uint8_t dimensional, ...)
{
  Multivector * multivector = NULL;

  ASSERT(0 <= dimensional);

  if (0 <= dimensional)
  {
    size_t memory_size = sizeof(Multivector) + (dimensional - 1) * sizeof(uint8_t);
    multivector = malloc(memory_size);

    ASSERT(multivector != NULL);

    if (multivector != NULL)
    {
      int arg;
      size_t data_size;
      va_list argument_list;

      memset(multivector, 0x00, memory_size);

      va_start(argument_list, dimensional);

      for (data_size = 1, arg = 0; arg < dimensional; arg ++)
        data_size *= (multivector->dimensional_size[arg] = (uint8_t) va_arg(argument_list, int));

      va_end(argument_list);

      multivector->data = ram_block_request(data_size * sizeof(float));

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size * sizeof(float));

      multivector->dimensional = dimensional;
    }
  }

  return multivector;
}

void Multivector_delete(Multivector ** multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(*multivector != NULL);

  if ((multivector != NULL) && (*multivector != NULL))
  {
    free(*multivector);
    *multivector = NULL;
  }
}

/*****************************************************************************/

SbsBaseLayer * SbsBaseLayer_new(uint16_t rows, uint16_t columns, uint16_t neurons)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);

  if (layer != NULL)
  {
    Multivector * state_matrix = NULL;

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    state_matrix = Multivector_new(3, rows, columns, neurons);

    layer->state_matrix = state_matrix;
  }

  return layer;
}

/*****************************************************************************/

Multivector * SbsNetwork_new(void)
{
  return Multivector_new(1);
}

void SbsNetwork_delete(Multivector ** sbsNetwork)
{
  Multivector_delete(sbsNetwork);
}

void SbsNetwork_addLayer()
{

}

/*****************************************************************************/

void sbs_test(void)
{
  Multivector * network = SbsNetwork_new();

  //Multivector * input_layer = SbsLayer_new();
}

/*****************************************************************************/
