/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: Yarib Nevarez
 */


//#define MULTIVECTOR_USE_POINTER_ARITHMETICS
//#define DEBUG


#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"
#include "stddef.h"
#include "stdarg.h"

#include "timer.h"
#include "miscellaneous.h"
#include "memory_manager.h"

#include "sbs_neural_network.h"
#include "mt19937int.h"


#include "ff.h"
#include "xparameters.h"

#include "xscugic.h"

#include "sbs_hardware_update.h"
#include "sbs_hardware_spike.h"
#include "dma_hardware_mover.h"

#include "sbs_processing_unit.h"

/*****************************************************************************/
#define   MEMORY_SIZE         (4771384)
#define   MAX_LAYER_SIZE      (28*28)
#define   MAX_KERNEL_SIZE     (5*5)
#define   MAX_IP_VECTOR_SIZE  (1024)  // Inference population size
#define   MAX_NETWORK_SIZE    (7)     // MAX number of layers in a network

/*****************************************************************************/

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */


typedef uint16_t  Weight;
typedef uint32_t  Random32;
typedef uint32_t  SpikeID;

typedef enum
{
  M32BIT_TYPE_BEGIN = 0,
  M_DOUBLE_1024_10_ID,
  M32BIT_24_24_ID,
  M32BIT_12_24_ID,
  M32BIT_24_24_50_ID,
  M32BIT_24_24_32_ID,
  M32BIT_12_24_32_ID,
  M32BIT_1_1_50_32_ID,
  M16BIT_1_1_50_32_ID,
  M32BIT_6_12_32_ID,
  M32BIT_12_12_32_ID,
  M32BIT_12_12_ID,
  M32BIT_6_12_ID,
  M32BIT_2_2_32_32_ID,
  M16BIT_2_2_32_32_ID,
  M32BIT_8_8_64_ID,
  M32BIT_4_8_64_ID,
  M32BIT_8_8_ID,
  M32BIT_4_8_ID,
  M32BIT_5_5_32_64_ID,
  M16BIT_5_5_32_64_ID,
  M32BIT_2_4_64_ID,
  M32BIT_4_4_64_ID,
  M32BIT_4_4_ID,
  M32BIT_2_4_ID,
  M32BIT_2_2_64_64_ID,
  M16BIT_2_2_64_64_ID,
  M32BIT_1_1_1024_ID,
  M32BIT_1_1_ID,
  M32BIT_4_4_64_1024_ID,
  M16BIT_4_4_64_1024_ID,
  M32BIT_1_1_10_ID,
  M32BIT_1_1_1024_10_ID,
  M16BIT_1_1_1024_10_ID,
  M32BIT_TYPE_END = (unsigned)-1
} MatrixTypeID;

typedef double   MDouble_1024_10[1024][10];
typedef uint32_t M32Bit_24_24[24][24];
typedef uint32_t M32Bit_12_24[12][24];
typedef uint32_t M32Bit_24_24_50[24][24][50];
typedef uint32_t M32Bit_12_24_32[12][24][32];
typedef uint32_t M32Bit_24_24_32[24][24][32];
typedef uint32_t M32Bit_1_1_50_32[1][1][50][32];
typedef Weight   M16Bit_1_1_50_32[1][1][50][32];
typedef uint32_t M32Bit_6_12_32[6][12][32];
typedef uint32_t M32Bit_12_12_32[6][12][32];
typedef uint32_t M32Bit_12_12[12][12];
typedef uint32_t M32Bit_6_12[6][12];
typedef uint32_t M32Bit_2_2_32_32[2][2][32][32];
typedef Weight   M16Bit_2_2_32_32[2][2][32][32];
typedef uint32_t M32Bit_8_8_64[8][8][64];
typedef uint32_t M32Bit_4_8_64[4][8][64];
typedef uint32_t M32Bit_8_8[8][8];
typedef uint32_t M32Bit_4_8[8][8];
typedef uint32_t M32Bit_5_5_32_64[5][5][32][64];
typedef Weight   M16Bit_5_5_32_64[5][5][32][64];
typedef uint32_t M32Bit_2_4_64[2][4][64];
typedef uint32_t M32Bit_4_4_64[4][4][64];
typedef uint32_t M32Bit_4_4[4][4];
typedef uint32_t M32Bit_2_4[2][4];
typedef uint32_t M32Bit_2_2_64_64[2][2][64][64];
typedef Weight   M16Bit_2_2_64_64[2][2][64][64];
typedef uint32_t M32Bit_1_1_1024[1][1][1024];
typedef uint32_t M32Bit_1_1[1][1];
typedef uint32_t M32Bit_4_4_64_1024[4][4][64][1024];
typedef Weight   M16Bit_4_4_64_1024[4][4][64][1024];
typedef uint32_t M32Bit_1_1_10[1][1][10];
typedef uint32_t M32Bit_1_1_1024_10[1][1][1024][10];
typedef Weight   M16Bit_1_1_1024_10[1][1][1024][10];

typedef struct
{
  MatrixTypeID type_id;
  uint8_t data_type_size;
  uint8_t dimensionality;
  uint16_t dimension_size[4];
} M32BitFormat;

M32BitFormat M32BitFormat_list[] =
{
    {
        .type_id = M_DOUBLE_1024_10_ID,
        .data_type_size = sizeof(double),
        .dimensionality = 2,
        .dimension_size = {1024, 10, 0, 0}
    },
    {
        .type_id = M32BIT_24_24_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {24, 24, 0, 0}
    },
    {
        .type_id = M32BIT_12_24_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {12, 24, 0, 0}
    },
    {
        .type_id = M32BIT_24_24_50_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {24, 24, 50, 0}
    },
    {
        .type_id = M32BIT_24_24_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {24, 24, 32, 0}
    },
    {
        .type_id = M32BIT_12_24_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {12, 24, 32, 0}
    },
    {
        .type_id = M32BIT_1_1_50_32_ID,
        .data_type_size = sizeof(float),
        .dimensionality = 4,
        .dimension_size = {1, 1, 50, 32}
    },
    {
        .type_id = M16BIT_1_1_50_32_ID,
        .data_type_size = sizeof(Weight),
        .dimensionality = 4,
        .dimension_size = {1, 1, 50, 32}
    },
    {
        .type_id = M32BIT_6_12_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {6, 12, 32, 0}
    },
    {
        .type_id = M32BIT_12_12_32_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {12, 12, 32, 0}
    },
    {
        .type_id = M32BIT_12_12_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {12, 12, 0, 0}
    },
    {
        .type_id = M32BIT_6_12_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {6, 12, 0, 0}
    },
    {
        .type_id = M32BIT_2_2_32_32_ID,
        .data_type_size = sizeof(float),
        .dimensionality = 4,
        .dimension_size = {2, 2, 32, 32}
    },
    {
        .type_id = M16BIT_2_2_32_32_ID,
        .data_type_size = sizeof(Weight),
        .dimensionality = 4,
        .dimension_size = {2, 2, 32, 32}
    },
    {
        .type_id = M32BIT_8_8_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {8, 8, 64, 0}
    },
    {
        .type_id = M32BIT_4_8_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {4, 8, 64, 0}
    },
    {
        .type_id = M32BIT_8_8_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {8, 8, 0, 0}
    },
    {
        .type_id = M32BIT_4_8_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {4, 8, 0, 0}
    },
    {
        .type_id = M32BIT_5_5_32_64_ID,
        .data_type_size = sizeof(float),
        .dimensionality = 4,
        .dimension_size = {5, 5, 32, 64}
    },
    {
        .type_id = M16BIT_5_5_32_64_ID,
        .data_type_size = sizeof(Weight),
        .dimensionality = 4,
        .dimension_size = {5, 5, 32, 64}
    },
    {
        .type_id = M32BIT_2_4_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {2, 4, 64, 0}
    },
    {
        .type_id = M32BIT_4_4_64_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {4, 4, 64, 0}
    },
    {
        .type_id = M32BIT_4_4_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {4, 4, 0, 0}
    },
    {
        .type_id = M32BIT_2_4_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {2, 4, 0, 0}
    },
    {
        .type_id = M32BIT_2_2_64_64_ID,
        .data_type_size = sizeof(float),
        .dimensionality = 4,
        .dimension_size = {2, 2, 64, 64}
    },
    {
        .type_id = M16BIT_2_2_64_64_ID,
        .data_type_size = sizeof(Weight),
        .dimensionality = 4,
        .dimension_size = {2, 2, 64, 64}
    },
    {
        .type_id = M32BIT_1_1_1024_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {1, 1, 1024, 0}
    },
    {
        .type_id = M32BIT_1_1_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 2,
        .dimension_size = {1, 1, 0, 0}
    },
    {
        .type_id = M32BIT_4_4_64_1024_ID,
        .data_type_size = sizeof(float),
        .dimensionality = 4,
        .dimension_size = {4, 4, 64, 1024}
    },
    {
        .type_id = M16BIT_4_4_64_1024_ID,
        .data_type_size = sizeof(Weight),
        .dimensionality = 4,
        .dimension_size = {4, 4, 64, 1024}
    },
    {
        .type_id = M32BIT_1_1_10_ID,
        .data_type_size = sizeof(uint32_t),
        .dimensionality = 3,
        .dimension_size = {1, 1, 10, 0}
    },
    {
        .type_id = M32BIT_1_1_1024_10_ID,
        .data_type_size = sizeof(float),
        .dimensionality = 4,
        .dimension_size = {1, 1, 1024, 10}
    },
    {
        .type_id = M16BIT_1_1_1024_10_ID,
        .data_type_size = sizeof(Weight),
        .dimensionality = 4,
        .dimension_size = {1, 1, 1024, 10}
    }
};

const unsigned M32BitFormat_list_length = (sizeof(M32BitFormat_list) / sizeof (M32BitFormat));

MatrixTypeID M32BitFormat_getTypeID(uint8_t data_type_size, uint8_t dimensionality, uint16_t * dimension_size)
{
  int i;
  MatrixTypeID type_ID = M32BIT_TYPE_END;

  for (i = 0; i < M32BitFormat_list_length; i++)
    if (M32BitFormat_list[i].data_type_size == data_type_size
        && M32BitFormat_list[i].dimensionality == dimensionality
        && 0 == memcmp (M32BitFormat_list[i].dimension_size,
                        dimension_size,
                        dimensionality * sizeof(uint16_t)))
      type_ID = M32BitFormat_list[i].type_id;

  ASSERT (type_ID != M32BIT_TYPE_END);

  return type_ID;
}

typedef struct
{
  MemoryBlock * memory_def_parent;

  void *   data;
  MatrixTypeID type_id;
  uint8_t  data_type_size;
  uint8_t  dimensionality;
  uint16_t dimension_size[1]; /*[0] = rows, [1] = columns, [2] = neurons... [n] = N*/
} Multivector;

typedef struct
{
  SbSUpdateAccelerator *  accelerator;
  SbsAcceleratorProfie    profile;
  uint16_t      x_pos;
  uint16_t      y_pos;
  Multivector * state_matrix;
  Multivector * spike_matrix;
  Multivector * weight_matrix;
} SbsLayerPartition;

typedef struct
{
  SbsLearningRule learning_rule;

  Multivector *   omega_matrix;
  Multivector *   a_matrix;
  Multivector *   b_matrix;
  float *         reco_vector;
  float *         delat_vector;
  double *        b_vector;

  unsigned int    number_of_patterns;
  unsigned int    current_pattern;
  double          gama;
} SbsLearningData;

typedef struct
{
  SbsLayer              vtbl;
  SbsLayerType          layer_type;
  SbsLayerPartition **  partition_array;
  uint8_t               num_partitions;
  uint16_t              rows;
  uint16_t              columns;
  uint16_t              vector_size;
  uint16_t              kernel_size;
  uint16_t              kernel_stride;
  WeightShift           weight_shift;
  Epsilon               epsilon;
  Multivector *         spike_matrix;
  SbsLearningData       learning_data;
} SbsBaseLayer;

typedef struct
{
  SbsNetwork        vtbl;
  uint8_t           size;
  SbsBaseLayer **   layer_array;
  uint8_t           input_label;
  uint8_t           inferred_output;
} SbsBaseNetwork;

#pragma pack(pop)   /* restore original alignment from stack */

static Multivector * Multivector_new(MemoryBlock * memory_def, uint8_t data_type_size, uint8_t dimensionality, ...)
{
  Multivector * multivector = NULL;

  ASSERT(0 <= dimensionality);

  if (0 <= dimensionality)
  {
    size_t memory_size = sizeof(Multivector) + (dimensionality - 1) * sizeof(uint16_t);
    multivector = malloc (memory_size);

    ASSERT(multivector != NULL);

    if (multivector != NULL)
    {
      int arg;
      size_t data_size;
      va_list argument_list;

      memset (multivector, 0x00, memory_size);

      va_start(argument_list, dimensionality);

      for (data_size = 1, arg = 0; arg < dimensionality; arg ++)
        data_size *= (multivector->dimension_size[arg] = (uint16_t) va_arg(argument_list, int));

      va_end(argument_list);

      multivector->memory_def_parent = memory_def;

      if (memory_def != NULL)
        multivector->data = MemoryBlock_alloc (memory_def,
                                               data_size * data_type_size);
      else
        multivector->data = malloc (data_size * data_type_size);

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size * data_type_size);

      multivector->dimensionality = dimensionality;
      multivector->data_type_size = data_type_size;
#ifndef MULTIVECTOR_USE_POINTER_ARITHMETICS
      multivector->type_id = M32BitFormat_getTypeID(data_type_size,
                                                    dimensionality,
                                                    multivector->dimension_size);
#endif
    }
  }

  return multivector;
}

//Multivector * multivector_array[80] = { 0 };
//int multivector_array_count = 0;
//
//void MultivectorArray_add(Multivector * multivector)
//{
//  int i;
//  for (i = 0;
//      i < multivector_array_count && multivector_array[i] != multivector;
//      i++);
//
//  if (i == multivector_array_count)
//  {
//    for (int t = 0; t < multivector_array_count; t ++)
//      if (multivector_array[t]->dimensionality == multivector->dimensionality)
//      {
//        int d;
//        for (d = 0; d < multivector_array[t]->dimensionality && (multivector_array[t]->dimension_size[d] == multivector->dimension_size[d]); d ++);
//
//        if (d == multivector_array[t]->dimensionality && multivector_array[t]->data_type_size == multivector->data_type_size)
//          return;
//      }
//
//    multivector_array[i] = multivector;
//    multivector_array_count ++;
//  }
//}
//
//int str_len (char * str)
//{
//  int i = 0;
//  while (str[i] != 0)
//    i++;
//  return i;
//}
//
//void MultivectorArray_print()
//{
//  int i;
//  int d;
//  char text[900] = {0};
//  for (i = 0; i < multivector_array_count; i++)
//  {
//    sprintf (&text[str_len(text)], "M[%d] = ", i);
//    for (d = 0; d < multivector_array[i]->dimensionality; d ++)
//      sprintf (&text[str_len(text)], "[%d]",multivector_array[i]->dimension_size[d]);
//    sprintf (&text[str_len(text)], "(%d)",multivector_array[i]->data_type_size);
//    sprintf (&text[str_len(text)], "\n");
//  }
//  printf ("Multivector catalog:\n%s\n",text);
//}

void inline * Multivector_2DAccess (Multivector * multivector,
                                    uint16_t row,
                                    uint16_t column) __attribute__((always_inline));

void inline * Multivector_3DAccess (Multivector * multivector,
                                    uint16_t row,
                                    uint16_t column,
                                    uint16_t position) __attribute__((always_inline));

#ifdef MULTIVECTOR_USE_POINTER_ARITHMETICS

void inline * Multivector_2DAccess(Multivector * multivector, uint16_t row, uint16_t column)
{
  void * data = NULL;
  ASSERT (multivector != NULL);
  ASSERT (multivector->data != NULL);
  ASSERT (2 <= multivector->dimensionality);
  ASSERT (row <= multivector->dimension_size[0]);
  ASSERT (column <= multivector->dimension_size[1]);

  //MultivectorArray_add(multivector);

  if ((multivector != NULL)
      && (multivector->data != NULL)
      && (2 <= multivector->dimensionality)
      && (row <= multivector->dimension_size[0])
      && (column <= multivector->dimension_size[1]))
  {
    uint16_t dimensionality = multivector->dimensionality;
    size_t data_size = multivector->data_type_size;

    while (dimensionality-- > 2)
    {
      data_size *= multivector->dimension_size[dimensionality];
    }

    data = multivector->data
        + (row * multivector->dimension_size[1] + column) * data_size;
  }

  return data;
}

void inline * Multivector_3DAccess (Multivector * multivector, uint16_t row, uint16_t column, uint16_t position)
{
  void * data = NULL;
  ASSERT (multivector != NULL);
  ASSERT (multivector->data != NULL);
  ASSERT (3 <= multivector->dimensionality);
  ASSERT (row <= multivector->dimension_size[0]);
  ASSERT (column <= multivector->dimension_size[1]);
  ASSERT (position <= multivector->dimension_size[2]);

  //MultivectorArray_add(multivector);

  if ((multivector != NULL)
      && (multivector->data != NULL)
      && (3 <= multivector->dimensionality)
      && (row <= multivector->dimension_size[0])
      && (column <= multivector->dimension_size[1])
      && (position <= multivector->dimension_size[2]))
  {
    uint16_t dimensionality = multivector->dimensionality;
    size_t data_size = multivector->data_type_size;

    while (dimensionality-- > 3)
    {
      data_size *= multivector->dimension_size[dimensionality];
    }

    data = multivector->data
        + ((row * multivector->dimension_size[1] + column)
            * multivector->dimension_size[2] + position) * data_size;
  }

  return data;
}

#else

void inline * Multivector_2DAccess (Multivector * multivector, uint16_t row, uint16_t column)
{
  ASSERT (multivector != NULL);
  ASSERT (multivector->data != NULL);
  ASSERT (2 <= multivector->dimensionality);
  ASSERT (row <= multivector->dimension_size[0]);
  ASSERT (column <= multivector->dimension_size[1]);

  switch (multivector->type_id)
  {
    case M_DOUBLE_1024_10_ID:
      return &(*(MDouble_1024_10*) multivector->data)[row][column];
    case M32BIT_24_24_ID:
      return &(*(M32Bit_24_24*) multivector->data)[row][column];
    case M32BIT_12_24_ID:
      return &(*(M32Bit_12_24*) multivector->data)[row][column];
    case M32BIT_24_24_50_ID:
      return &(*(M32Bit_24_24_50*) multivector->data)[row][column];
    case M32BIT_12_24_32_ID:
      return &(*(M32Bit_12_24_32*) multivector->data)[row][column];
    case M32BIT_24_24_32_ID:
      return &(*(M32Bit_24_24_32*) multivector->data)[row][column];
    case M32BIT_1_1_50_32_ID:
      return &(*(M32Bit_1_1_50_32*) multivector->data)[row][column];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column];
    case M32BIT_12_12_32_ID:
      return &(*(M32Bit_12_12_32*) multivector->data)[row][column];
    case M32BIT_12_12_ID:
      return &(*(M32Bit_12_12*) multivector->data)[row][column];
    case M32BIT_6_12_ID:
      return &(*(M32Bit_6_12*) multivector->data)[row][column];
    case M32BIT_2_2_32_32_ID:
      return &(*(M32Bit_2_2_32_32*) multivector->data)[row][column];
    case M32BIT_8_8_64_ID:
      return &(*(M32Bit_8_8_64*) multivector->data)[row][column];
    case M32BIT_4_8_64_ID:
      return &(*(M32Bit_4_8_64*) multivector->data)[row][column];
    case M32BIT_8_8_ID:
      return &(*(M32Bit_8_8*) multivector->data)[row][column];
    case M32BIT_5_5_32_64_ID:
      return &(*(M32Bit_5_5_32_64*) multivector->data)[row][column];
    case M32BIT_2_4_64_ID:
      return &(*(M32Bit_2_4_64*) multivector->data)[row][column];
    case M32BIT_4_4_64_ID:
      return &(*(M32Bit_4_4_64*) multivector->data)[row][column];
    case M32BIT_4_4_ID:
      return &(*(M32Bit_4_4*) multivector->data)[row][column];
    case M32BIT_2_2_64_64_ID:
      return &(*(M32Bit_2_2_64_64*) multivector->data)[row][column];
    case M32BIT_1_1_1024_ID:
      return &(*(M32Bit_1_1_1024*) multivector->data)[row][column];
    case M32BIT_1_1_ID:
      return &(*(M32Bit_1_1*) multivector->data)[row][column];
    case M32BIT_4_4_64_1024_ID:
      return &(*(M32Bit_4_4_64_1024*) multivector->data)[row][column];
    case M32BIT_1_1_10_ID:
      return &(*(M32Bit_1_1_10*) multivector->data)[row][column];
    case M32BIT_1_1_1024_10_ID:
      return &(*(M32Bit_1_1_1024_10*) multivector->data)[row][column];
    default:
      ASSERT (0);
  }
  return NULL;
}

void inline * Multivector_3DAccess (Multivector * multivector, uint16_t row, uint16_t column, uint16_t position)
{
  ASSERT(multivector != NULL);
  ASSERT(multivector->data != NULL);
  ASSERT(3 <= multivector->dimensionality);
  ASSERT(row <= multivector->dimension_size[0]);
  ASSERT(column <= multivector->dimension_size[1]);
  ASSERT(position <= multivector->dimension_size[2]);

  switch (multivector->type_id)
  {
    case M32BIT_24_24_50_ID:
      return &(*(M32Bit_24_24_50*) multivector->data)[row][column][position];
    case M32BIT_12_24_32_ID:
      return &(*(M32Bit_12_24_32*) multivector->data)[row][column][position];
    case M32BIT_1_1_50_32_ID:
      return &(*(M32Bit_1_1_50_32*) multivector->data)[row][column][position];
    case M16BIT_1_1_50_32_ID:
        return &(*(M16Bit_1_1_50_32*) multivector->data)[row][column][position];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column][position];
    case M32BIT_2_2_32_32_ID:
      return &(*(M32Bit_2_2_32_32*) multivector->data)[row][column][position];
    case M16BIT_2_2_32_32_ID:
        return &(*(M16Bit_2_2_32_32*) multivector->data)[row][column][position];
    case M32BIT_8_8_64_ID:
      return &(*(M32Bit_8_8_64*) multivector->data)[row][column][position];
    case M32BIT_5_5_32_64_ID:
      return &(*(M32Bit_5_5_32_64*) multivector->data)[row][column][position];
    case M16BIT_5_5_32_64_ID:
      return &(*(M16Bit_5_5_32_64*) multivector->data)[row][column][position];
    case M32BIT_2_4_64_ID:
      return &(*(M32Bit_2_4_64*) multivector->data)[row][column][position];
    case M32BIT_2_2_64_64_ID:
      return &(*(M32Bit_2_2_64_64*) multivector->data)[row][column][position];
    case M16BIT_2_2_64_64_ID:
      return &(*(M16Bit_2_2_64_64*) multivector->data)[row][column][position];
    case M32BIT_1_1_1024_ID:
      return &(*(M32Bit_1_1_1024*) multivector->data)[row][column][position];
    case M32BIT_4_4_64_1024_ID:
      return &(*(M32Bit_4_4_64_1024*) multivector->data)[row][column][position];
    case M16BIT_4_4_64_1024_ID:
      return &(*(M16Bit_4_4_64_1024*) multivector->data)[row][column][position];
    case M32BIT_1_1_10_ID:
      return &(*(M32Bit_1_1_10*) multivector->data)[row][column][position];
    case M32BIT_1_1_1024_10_ID:
      return &(*(M32Bit_1_1_1024_10*) multivector->data)[row][column][position];
    case M16BIT_1_1_1024_10_ID:
      return &(*(M16Bit_1_1_1024_10*) multivector->data)[row][column][position];
    default:
      ASSERT(0)
      ;
  }
  return NULL;
}
#endif

static Multivector * Multivector_duplicate(MemoryBlock * memory_def,
                                           Multivector * original)
{
  Multivector * duplicate = NULL;
  ASSERT(original != NULL);
  ASSERT(0 < original->dimensionality);

  if ((original != NULL)
      && (0 < original->dimensionality))
  {
    size_t memory_size = sizeof(Multivector)
        + (original->dimensionality - 1) * sizeof(uint16_t);
    duplicate = malloc (memory_size);

    ASSERT(duplicate != NULL);

    if (duplicate != NULL)
    {
      size_t data_size = original->data_type_size;
      int i;

      memcpy (duplicate, original, memory_size);

      for (i = 0; i < original->dimensionality; i++)
        data_size *= original->dimension_size[i];

      duplicate->memory_def_parent = memory_def;

      if (memory_def != NULL)
        duplicate->data = MemoryBlock_alloc (memory_def, data_size);
      else
        duplicate->data = malloc (data_size);

      ASSERT(duplicate->data != NULL);

      if (duplicate->data != NULL)
        memcpy (duplicate->data, original->data, data_size);
      else
        return NULL;
    }
  }
  return duplicate;
}

static size_t Multivector_dataSize(Multivector * multivector)
{
  size_t data_size = 0;
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    int i;
    data_size = multivector->data_type_size;

    for (i = 0; i < multivector->dimensionality; i++)
      data_size *= multivector->dimension_size[i];
  }
  return data_size;
}

static void Multivector_cacheFlush(Multivector * multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    size_t data_size = multivector->data_type_size;
    int i;

    for (i = 0; i < multivector->dimensionality; i++)
      data_size *= multivector->dimension_size[i];

    Xil_DCacheFlushRange ((UINTPTR) multivector->data, data_size);
  }
}

static void Multivector_delete(Multivector ** multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(*multivector != NULL);

  if ((multivector != NULL) && (*multivector != NULL))
  {
    if ((*multivector)->memory_def_parent == NULL)
      free ((*multivector)->data);

    free(*multivector);
    *multivector = NULL;
  }
}

static void Multivector_float2Fixed(Multivector * multivector)
{
  ASSERT(multivector != NULL);

  if (multivector != NULL)
  {
    int size = Multivector_dataSize(multivector) / multivector->data_type_size;
	  float *     float_ptr = (float *)     multivector->data;
	  uint32_t *  fixed_ptr = (uint32_t *)  multivector->data;
    for (int i = 0; i < size; i ++)
    {
      fixed_ptr[i] = (uint32_t) (float_ptr[i] * H_MAX);
    }
  }
}

static Multivector * Multivector_convert2fixed16(MemoryBlock * memory_def,
                                                 Multivector * original)
{
  Multivector * duplicate = NULL;
  ASSERT(original != NULL);
  ASSERT(0 < original->dimensionality);
  ASSERT(original->data_type_size == sizeof(uint32_t));

  if ((original != NULL)
      && (0 < original->dimensionality))
  {
    size_t memory_size = sizeof(Multivector)
        + (original->dimensionality - 1) * sizeof(uint16_t);
    duplicate = malloc (memory_size);

    ASSERT(duplicate != NULL);

    if (duplicate != NULL)
    {
      size_t matrix_size = 1;
      int i;

      memcpy (duplicate, original, memory_size);

      duplicate->data_type_size = sizeof(uint16_t);

      for (i = 0; i < original->dimensionality; i++)
        matrix_size *= original->dimension_size[i];

      duplicate->memory_def_parent = memory_def;

      if (memory_def != NULL)
        duplicate->data = MemoryBlock_alloc (memory_def, matrix_size * sizeof(uint16_t));
      else
        duplicate->data = malloc (matrix_size * sizeof(uint16_t));

      ASSERT(duplicate->data != NULL);

      if (duplicate->data != NULL)
      {
        float *    original_data  = original->data;
        uint16_t * duplicate_data = duplicate->data;

        for (int i = 0; i < matrix_size; i ++)
          duplicate_data[i] = (uint16_t)(original_data[i] * (float)(0xFFFF));

#ifndef MULTIVECTOR_USE_POINTER_ARITHMETICS
        duplicate->type_id = M32BitFormat_getTypeID (duplicate->data_type_size,
                                                     duplicate->dimensionality,
                                                     duplicate->dimension_size);
#endif

      }
      else
      {
        free (duplicate);
        return NULL;
      }
    }
  }
  return duplicate;
}

/*****************************************************************************/
void SbsAcceleratorProfie_initialize(SbsAcceleratorProfie * profile,
                                     SbsLayerType layerType,
                                     Multivector * state_matrix,
                                     Multivector * weight_matrix,
                                     Multivector * spike_matrix,
                                     uint32_t kernel_size,
                                     Epsilon epsilon,
                                     MemoryCmd memory_cmd)
{
  ASSERT (profile != NULL);
  ASSERT (state_matrix != NULL);
  ASSERT (state_matrix->dimensionality == 3);
  ASSERT (state_matrix->data != NULL);

  if ((profile != NULL)
      && (state_matrix != NULL)
      && (state_matrix->dimensionality == 3)
      && (state_matrix->data != NULL))
  {
    profile->layerSize =
        state_matrix->dimension_size[0] * state_matrix->dimension_size[1];
    profile->vectorSize = state_matrix->dimension_size[2];

    profile->kernelSize = kernel_size * kernel_size;
    profile->epsilon = epsilon;

    profile->stateBufferSize = profile->vectorSize * state_matrix->data_type_size;

    /**************************** SPIKE_MODE *********************************/
    profile->rxBuffer[SPIKE_MODE] = spike_matrix->data;
    profile->rxBufferSize[SPIKE_MODE] = Multivector_dataSize(spike_matrix);
    profile->txBufferSize[SPIKE_MODE] = profile->layerSize
        * (sizeof(Random32) + profile->vectorSize * state_matrix->data_type_size);

    ASSERT (profile->txBuffer[SPIKE_MODE] == NULL);
    profile->txBuffer[SPIKE_MODE] = MemoryBlock_alloc(state_matrix->memory_def_parent, profile->txBufferSize[SPIKE_MODE]);
    ASSERT (profile->txBuffer[SPIKE_MODE] != NULL);

    /**************************** UPDATE_MODE ********************************/
    if (weight_matrix != NULL)
    {
      ASSERT(weight_matrix->dimension_size[3] == state_matrix->dimension_size[2]);

      profile->weightBufferSize = profile->vectorSize * weight_matrix->data_type_size;

      profile->memory_cmd[UPDATE_MODE] = memory_cmd;
      profile->rxBuffer[UPDATE_MODE] = state_matrix->data;
      profile->rxBufferSize[UPDATE_MODE] = Multivector_dataSize(state_matrix)+ Multivector_dataSize(spike_matrix);
      profile->txBufferSize[UPDATE_MODE] = profile->layerSize
      * (sizeof(Random32) + profile->vectorSize * state_matrix->data_type_size + profile->kernelSize * profile->vectorSize * weight_matrix->data_type_size);

      ASSERT (profile->txBuffer[UPDATE_MODE] == NULL);
      profile->txBuffer[UPDATE_MODE] = MemoryBlock_alloc(state_matrix->memory_def_parent, profile->txBufferSize[UPDATE_MODE]);
      ASSERT (profile->txBuffer[UPDATE_MODE] != NULL);
    }
  }
}

static uint8_t SbsAcceleratorProfie_isInitialized(SbsAcceleratorProfie * profile)
{
  return (profile->txBuffer[SPIKE_MODE] != NULL);
}

/*****************************************************************************/
static SbsLayerPartition * SbsLayerPartition_new (SbSUpdateAccelerator * accelerator,
                                                  uint16_t x_pos,
                                                  uint16_t y_pos,
                                                  uint16_t rows,
                                                  uint16_t columns,
                                                  uint16_t vector_size)
{
  SbsLayerPartition * partition = NULL;

  partition = (SbsLayerPartition *) malloc (sizeof(SbsLayerPartition));

  ASSERT (partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = NULL;
    Multivector * spike_matrix = NULL;
    MemoryBlock * memory_def = NULL;

    memset (partition, 0x00, sizeof(SbsLayerPartition));

    if (accelerator != NULL)
    {
      ASSERT (accelerator->hardwareConfig != NULL);

      if (accelerator->hardwareConfig != NULL)
      {
        partition->accelerator = accelerator;
        memory_def = &accelerator->hardwareConfig->ddrMem;
      }
    }

    /* Instantiate state_matrix */
    state_matrix = Multivector_new (memory_def, sizeof(NeuronState), 3, rows,
                                    columns, vector_size);

    ASSERT(state_matrix != NULL);
    ASSERT(state_matrix->dimensionality == 3);
    ASSERT(state_matrix->data != NULL);
    ASSERT(state_matrix->dimension_size[0] == rows);
    ASSERT(state_matrix->dimension_size[1] == columns);
    ASSERT(state_matrix->dimension_size[2] == vector_size);

    partition->state_matrix = state_matrix;

    /* Instantiate spike_matrix */
    spike_matrix = Multivector_new (memory_def, sizeof(SpikeID), 2, rows, columns);

    ASSERT(spike_matrix != NULL);
    ASSERT(spike_matrix->dimensionality == 2);
    ASSERT(spike_matrix->data != NULL);
    ASSERT(spike_matrix->dimension_size[0] == rows);
    ASSERT(spike_matrix->dimension_size[1] == columns);

    partition->spike_matrix = spike_matrix;

    partition->x_pos = x_pos;
    partition->y_pos = y_pos;
  }

  return partition;
}

static void SbsLayerPartition_delete(SbsLayerPartition ** partition)
{
  ASSERT (partition != NULL);
  ASSERT (*partition != NULL);

  if ((partition != NULL) && (*partition != NULL))
  {
    Multivector_delete (&((*partition)->state_matrix));
    Multivector_delete (&((*partition)->spike_matrix));
    if ((*partition)->weight_matrix != NULL)
      Multivector_delete (&((*partition)->weight_matrix));

    free (*partition);

    *partition = NULL;
  }
}

static void SbsLayerPartition_initializeIP (NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
	  NeuronState	initial_value_h = H_MAX / size;
      uint16_t 		neuron;
      for (neuron = 0; neuron < size; neuron ++)
        state_vector[neuron] = initial_value_h;
  }
}

static void SbsLayerPartition_initialize (SbsLayerPartition * partition,
                                          SbsLayerType layerType,
                                          uint32_t kernel_size,
                                          Epsilon epsilon,
                                          MemoryCmd accelerator_memory_cmd)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector * state_matrix = partition->state_matrix;
    uint16_t rows = state_matrix->dimension_size[0];
    uint16_t columns = state_matrix->dimension_size[1];
    uint16_t neurons = state_matrix->dimension_size[2];

    uint16_t row;
    uint16_t column;

    if (layerType != HX_INPUT_LAYER)
      for (row = 0; row < rows; row++)
        for (column = 0; column < columns; column++)
          SbsLayerPartition_initializeIP (Multivector_2DAccess(state_matrix, row, column), neurons);

    if (!SbsAcceleratorProfie_isInitialized (&partition->profile))
      SbsAcceleratorProfie_initialize (&partition->profile,
                                       layerType,
                                       state_matrix,
                                       partition->weight_matrix,
                                       partition->spike_matrix,
                                       kernel_size,
                                       epsilon,
                                       accelerator_memory_cmd);
  }
}

static void SbsLayerPartition_cacheFlush (SbsLayerPartition * partition)
{
  ASSERT(partition != NULL);

  if (partition != NULL)
  {
    Multivector_cacheFlush (partition->state_matrix);

    if (partition->weight_matrix != NULL)
      Multivector_cacheFlush (partition->weight_matrix);
  }
}

static void SbsLayerPartition_setWeights (SbsLayerPartition * partition,
                                          SbsWeightMatrix weight_matrix)
{
  ASSERT(partition != NULL);
  ASSERT(partition->accelerator != NULL);
  ASSERT(partition->accelerator->hardwareConfig != NULL);
  ASSERT(weight_matrix != NULL);

  if ((partition != NULL)
      && (partition->accelerator != NULL)
      && (partition->accelerator->hardwareConfig != NULL)
      && (weight_matrix != NULL))
  {
    if (partition->weight_matrix != NULL)
      Multivector_delete(&partition->weight_matrix);

    partition->weight_matrix =
        Multivector_duplicate(&partition->accelerator->hardwareConfig->ddrMem,
                              weight_matrix);
  }
}

/*****************************************************************************/

static SbsLayer * SbsBaseLayer_new(SbsLayerType layer_type,
                                   uint16_t rows,
                                   uint16_t columns,
                                   uint16_t neurons,
                                   uint16_t kernel_size,
                                   uint16_t kernel_stride,
                                   WeightShift weight_shift)
{
  SbsBaseLayer * layer = malloc(sizeof(SbsBaseLayer));

  ASSERT(layer != NULL);
  ASSERT(rows * columns <= 60 * 60);
  ASSERT(neurons <= 1024);

  if (layer != NULL)
  {
    int i;
    SbSUpdateAccelerator * accelerator_group[/*NUM_ACCELERATOR_INSTANCES*/10] = {0};

    memset(layer, 0x00, sizeof(SbsBaseLayer));

    layer->vtbl = _SbsLayer;

    layer->layer_type = layer_type;
    layer->num_partitions = SbSUpdateAccelerator_getGroupFromList (
        layer_type, accelerator_group, /*NUM_ACCELERATOR_INSTANCES*/10);

    ASSERT (0 < layer->num_partitions);

    layer->partition_array = (SbsLayerPartition **)
        malloc (layer->num_partitions * sizeof(SbsLayerPartition *));
    ASSERT(layer->partition_array != NULL);

    if (layer->partition_array != NULL)
    {
      uint16_t residual = (rows % layer->num_partitions);
      uint16_t rows_per_partition = ((rows - residual) / layer->num_partitions);
      uint16_t rows_current_partition;
      uint16_t pos_y = 0;
      uint16_t pos_x = 0;

      ASSERT (((rows - residual) % layer->num_partitions) == 0);

      SbSUpdateAccelerator * accelerator = NULL;
      for (i = 0; i < layer->num_partitions; i++)
      {
        accelerator = accelerator_group[i];

        if (0 < residual)
        {
          rows_current_partition = rows_per_partition + 1;
          residual--;
        }
        else
        {
          rows_current_partition = rows_per_partition;
        }

        layer->partition_array[i] = SbsLayerPartition_new (accelerator,
                                                           pos_x,
                                                           pos_y,
                                                           rows_current_partition,
                                                           columns,
                                                           neurons);

        pos_y += rows_current_partition;
      }
    }

    if (1 < layer->num_partitions)
    { /* Instantiate spike_matrix */
      Multivector * spike_matrix = Multivector_new (NULL, sizeof(SpikeID), 2, rows, columns);

      ASSERT(spike_matrix != NULL);
      ASSERT(spike_matrix->dimensionality == 2);
      ASSERT(spike_matrix->data != NULL);
      ASSERT(spike_matrix->dimension_size[0] == rows);
      ASSERT(spike_matrix->dimension_size[1] == columns);

      layer->spike_matrix = spike_matrix;
    }
    else
      layer->spike_matrix = layer->partition_array[0]->spike_matrix;

    /* Assign parameters */
    layer->rows          = rows;
    layer->columns       = columns;
    layer->vector_size   = neurons;
    layer->kernel_size   = kernel_size;
    layer->kernel_stride = kernel_stride;
    layer->weight_shift  = weight_shift;
  }

  return (SbsLayer *) layer;
}

static void SbsBaseLayer_delete(SbsLayer ** layer_ptr)
{
  ASSERT(layer_ptr!= NULL);
  ASSERT(*layer_ptr!= NULL);
  if ((layer_ptr != NULL) && (*layer_ptr != NULL))
  {
    SbsBaseLayer ** layer = (SbsBaseLayer **) layer_ptr;

    if ((*layer)->spike_matrix->memory_def_parent == NULL)
      Multivector_delete(&((*layer)->spike_matrix));

    if ((*layer)->partition_array != NULL)
      while ((*layer)->num_partitions)
        SbsLayerPartition_delete (&((*layer)->partition_array[--(*layer)->num_partitions]));

    free (*layer);
    *layer = NULL;
  }
}

static void SbsBaseLayer_initialize(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    MemoryCmd accelerator_memory_cmd = { .cmdID = MEM_CMD_NONE };
    int i;

    if (1 < layer->num_partitions)
    {
      accelerator_memory_cmd.cmdID = MEM_CMD_MOVE;
      accelerator_memory_cmd.dest = layer->spike_matrix->data;
    }

    for (i = 0; i < layer->num_partitions; i++)
    {
      accelerator_memory_cmd.src = layer->partition_array[i]->spike_matrix->data;
      accelerator_memory_cmd.size = Multivector_dataSize(layer->partition_array[i]->spike_matrix);

      SbsLayerPartition_initialize (layer->partition_array[i],
                                    layer->layer_type,
                                    layer->kernel_size,
                                    layer->epsilon,
                                    accelerator_memory_cmd);

      accelerator_memory_cmd.dest += accelerator_memory_cmd.size;
    }
  }
}

static void SbsBaseLayer_cacheFlush(SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(0 < layer->num_partitions);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (0 < layer->num_partitions))
  {
    int i;
    for (i = 0; i < layer->num_partitions; i++)
      SbsLayerPartition_cacheFlush(layer->partition_array[i]);
  }
}

static void SbsBaseLayer_giveWeights(SbsLayer * layer, SbsWeightMatrix weight_matrix)
{
  ASSERT(layer != NULL);
  ASSERT(weight_matrix != NULL);

  if ((layer != NULL) && (((SbsBaseLayer *) layer)->partition_array != NULL))
  {
    int i;

    for (i = 0; i < ((SbsBaseLayer *) layer)->num_partitions; i++)
      SbsLayerPartition_setWeights (((SbsBaseLayer *) layer)->partition_array[i],
                                     weight_matrix);

    Multivector_delete ((Multivector**) &weight_matrix);
  }
}

static void SbsBaseLayer_setEpsilon(SbsLayer * layer, float epsilon)
{
  ASSERT(layer != NULL);
  /*ASSERT(epsilon != 0.0f);*/ /* 0.0 is allowed? */

  if (layer != NULL)
    ((SbsBaseLayer *)layer)->epsilon = (Epsilon) (epsilon * H_MAX);
}

static void SbsBaseLayer_setLearningRule (SbsLayer * layer_ptr, SbsLearningRule rule, double gama, int number_of_patterns)
{
  ASSERT (layer_ptr != NULL);
  ASSERT (0 < number_of_patterns);
  if ((layer_ptr != NULL) && (0 < number_of_patterns))
  {
    SbsBaseLayer * layer = (SbsBaseLayer *) layer_ptr;
    layer->learning_data.learning_rule = rule;
    layer->learning_data.number_of_patterns = number_of_patterns;
    layer->learning_data.current_pattern = 0;
    int w_spikes = layer->partition_array[0]->weight_matrix->dimension_size[2];

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.omega_matrix != NULL)
      Multivector_delete(&layer->learning_data.omega_matrix);

    layer->learning_data.omega_matrix = Multivector_new(NULL, sizeof(double), 2, w_spikes, layer->vector_size);

    ASSERT (layer->learning_data.omega_matrix != NULL);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.a_matrix != NULL)
      Multivector_delete(&layer->learning_data.a_matrix);

    layer->learning_data.a_matrix = Multivector_new(NULL, sizeof(double), 2, w_spikes, layer->vector_size);

    ASSERT (layer->learning_data.a_matrix != NULL);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.b_matrix != NULL)
      Multivector_delete(&layer->learning_data.b_matrix);

    layer->learning_data.b_matrix = Multivector_new(NULL, sizeof(double), 2, w_spikes, layer->vector_size);

    ASSERT (layer->learning_data.b_matrix != NULL);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.reco_vector != NULL)
      free(layer->learning_data.reco_vector);

    layer->learning_data.reco_vector = malloc(sizeof(float) * w_spikes);

    ASSERT (layer->learning_data.reco_vector != NULL);

    memset (layer->learning_data.reco_vector, 0, sizeof(float) * w_spikes);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.delat_vector != NULL)
      free(layer->learning_data.delat_vector);

    layer->learning_data.delat_vector = malloc(sizeof(float) * w_spikes);

    ASSERT (layer->learning_data.delat_vector != NULL);

    memset (layer->learning_data.delat_vector, 0, sizeof(float) * w_spikes);

    ///////////////////////////////////////////////////////////////////////////

    if (layer->learning_data.b_vector != NULL)
      free(layer->learning_data.b_vector);

    layer->learning_data.b_vector = malloc(sizeof(double) * layer->vector_size);

    ASSERT (layer->learning_data.b_vector != NULL);

    memset (layer->learning_data.b_vector, 0, sizeof(double) * layer->vector_size);

    ///////////////////////////////////////////////////////////////////////////

    layer->learning_data.gama = gama;
  }
}

inline static SpikeID SbsStateVector_generateSpike (NeuronState * state_vector, uint16_t size) __attribute__((always_inline));

inline static SpikeID SbsStateVector_generateSpike (NeuronState * state_vector, uint16_t size)
{
  ASSERT(state_vector != NULL);
  ASSERT(0 < size);

  if ((state_vector != NULL) && (0 < size))
  {
    NeuronState random_s = ((NeuronState) genrand ()) >> (32 - H_QF);
    NeuronState sum      = 0;
    SpikeID     spikeID;

    ASSERT(random_s <= H_MAX);

    for (spikeID = 0; spikeID < size; spikeID++)
    {
      sum += state_vector[spikeID];

      //ASSERT(sum <= 1 + 1e-5);

      if (random_s <= sum)
        return spikeID;
    }
  }

  return size - 1;
}

static void SbsLayerPartition_loadInput(SbsLayerPartition * partition, char * file_name, uint8_t * input_label)
{
  ASSERT(partition != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  if ((partition != NULL) && (file_name != NULL) && (input_label != NULL))
  {
    FIL fil; /* File object */
    FRESULT rc;
    rc = f_open (&fil, file_name, FA_READ);
    ASSERT(rc == FR_OK);

    if (rc == FR_OK)
    {
      uint16_t       rows        = partition->state_matrix->dimension_size[0];
      uint16_t       columns     = partition->state_matrix->dimension_size[1];
      uint16_t       neurons     = partition->state_matrix->dimension_size[2];
      NeuronState *  data        = partition->state_matrix->data;

      uint16_t row;
      uint16_t column;
      size_t   read_result = 0;

      uint8_t good_reading_flag = 1;

      size_t inference_population_size = sizeof(NeuronState) * neurons;

      for (column = 0; (column < columns) && good_reading_flag; column++)
        for (row = 0; (row < rows) && good_reading_flag; row++)
        {
          rc = f_read (&fil, &data[column * neurons + row * columns * neurons],
                       inference_population_size, &read_result);

          good_reading_flag = read_result == inference_population_size;
        }

      if (good_reading_flag)
      {
        rc = f_read (&fil, input_label, sizeof(uint8_t), &read_result);
        (*input_label)--;
        good_reading_flag = read_result == sizeof(uint8_t);

        Multivector_float2Fixed(partition->state_matrix);
      }

      f_close (&fil);
      ASSERT(good_reading_flag);
    }

  }
}

static void SbsBaseLayer_loadInput (SbsBaseLayer * layer, char * file_name,
                                    uint8_t * input_label)
{
  ASSERT(layer != NULL);
  ASSERT(file_name != NULL);
  ASSERT(input_label != NULL);
  ASSERT(layer->layer_type == HX_INPUT_LAYER);
  ASSERT(layer->num_partitions == 1);
  if ((layer != NULL) && (file_name != NULL) && (input_label != NULL))
  {
    SbsLayerPartition_loadInput (layer->partition_array[0], file_name,
                                 input_label);
  }
}

static void SbsBaseLayer_getOutputVector(SbsBaseLayer * layer,
                                         NeuronState ** output_vector,
                                         uint16_t * output_vector_size)
{
  ASSERT(layer != NULL);
  ASSERT(0 < layer->num_partitions);
  ASSERT(layer->layer_type == HY_OUTPUT_LAYER);
  ASSERT(layer->partition_array[layer->num_partitions - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != NULL);

  if ((layer != NULL) && (0 < layer->num_partitions)
      && (layer->layer_type == HY_OUTPUT_LAYER)
      && (layer->partition_array[layer->num_partitions - 1] != NULL)
      && (output_vector != NULL) && (output_vector_size != NULL))
  {
    SbsLayerPartition *  partition = layer->partition_array[layer->num_partitions - 1];
    Multivector * output_state_matrix = partition->state_matrix;

    ASSERT(output_state_matrix->data != NULL);
    ASSERT(output_state_matrix->dimensionality == 3);
    ASSERT(output_state_matrix->dimension_size[0] == 1);
    ASSERT(output_state_matrix->dimension_size[1] == 1);
    ASSERT(0 < output_state_matrix->dimension_size[2]);

    * output_vector = output_state_matrix->data;
    * output_vector_size = output_state_matrix->dimension_size[2];
  }
}

inline SbsLayerPartition * SbsBaseLayer_getPartition(SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                              uint16_t * partition_row, uint16_t * partition_column) __attribute__((always_inline));

inline SbsLayerPartition * SbsBaseLayer_getPartition(SbsBaseLayer * layer, uint16_t row, uint16_t column,
                                              uint16_t * partition_row, uint16_t * partition_column)
{
  SbsLayerPartition * partition = NULL;
  if (layer->num_partitions == 1)
  {
    partition = layer->partition_array[0];
    if (partition_row) *partition_row = row;
    if (partition_column) *partition_column = column;
  }
  else
  {
    int i;
    for (i = 0; partition == NULL && i < (layer)->num_partitions; i++)
    {
      if (layer->partition_array[i]->x_pos <= column
          && column < layer->partition_array[i]->x_pos + layer->partition_array[i]->state_matrix->dimension_size[1]
          && layer->partition_array[i]->y_pos <= row
          && row < layer->partition_array[i]->y_pos + layer->partition_array[i]->state_matrix->dimension_size[0])
      {
        partition = layer->partition_array[i];
        if (partition_row) *partition_row = row - layer->partition_array[i]->y_pos;
        if (partition_column) *partition_column = column - layer->partition_array[i]->x_pos;
      }
    }
  }

  return partition;
}

//static void SbsBaseLayer_generateSpikes (SbsBaseLayer * layer)
//{
//  ASSERT(layer != NULL);
//  ASSERT(layer->partition_array != NULL);
//  ASSERT(0 < layer->num_partitions);
//  ASSERT(layer->spike_matrix != NULL);
//
//  if ((layer != NULL)
//      && (layer->partition_array != NULL)
//      && (0 < layer->num_partitions)
//      && (layer->spike_matrix != NULL))
//  {
//    int i;
//    uint16_t columns = layer->columns;
//    uint16_t neurons = layer->vector_size;
//
//    uint16_t partition_row = 0;
//    SbsLayerPartition *  partition = NULL;
//    Multivector * partition_state_matrix = NULL;
//    Multivector* layer_spike_matrix = layer->spike_matrix;
//
//    SpikeID * spike;
//    NeuronState * state_vector;
//
//    uint16_t row;
//    uint16_t column;
//
//    for (i = 0; i < layer->num_partitions; i ++)
//    {
//      ASSERT(layer->partition_array[i] != NULL);
//      ASSERT(layer->partition_array[i]->state_matrix != NULL);
//      partition = layer->partition_array[i];
//      partition_state_matrix = partition->state_matrix;
//
//      for (row = partition->y_pos, partition_row = 0;
//          partition_row < partition_state_matrix->dimension_size[0];
//          partition_row++, row ++)
//      {
//        for (column = 0; column < columns; column++)
//        {
//          spike = Multivector_2DAccess (layer_spike_matrix, row, column);
//          state_vector = Multivector_2DAccess (partition_state_matrix,
//                                               partition_row,
//                                               column);
//          *spike = SbsStateVector_generateSpike (state_vector, neurons);
//        }
//      }
//    }
//  }
//}

static void SbsBaseLayer_generateSpikes (SbsBaseLayer * layer)
{
  ASSERT(layer != NULL);
  ASSERT(layer->partition_array != NULL);
  ASSERT(1 == layer->num_partitions);
  ASSERT(layer->layer_type == HX_INPUT_LAYER);

  if ((layer != NULL)
      && (layer->partition_array != NULL)
      && (1 == layer->num_partitions))
  {
    SbsLayerPartition * partition     = layer->partition_array[0];
    Multivector *       state_matrix  = partition->state_matrix;
    uint16_t            columns       = layer->columns;
    uint16_t            rows          = layer->rows;
    uint16_t            row;
    uint16_t            column;

    ASSERT (layer->partition_array[0] != NULL);
    ASSERT (layer->partition_array[0]->state_matrix != NULL);
    ASSERT (layer->partition_array[0]->spike_matrix != NULL);
    ASSERT (rows == state_matrix->dimension_size[0]);
    ASSERT (columns == state_matrix->dimension_size[1]);

    Accelerator_setup (partition->accelerator, &partition->profile, SPIKE_MODE);

    for (row = 0; row < rows; row++)
      for (column = 0; column < columns; column++)
        Accelerator_giveStateVector (partition->accelerator,
                                     Multivector_2DAccess (state_matrix, row, column));

    Accelerator_start (partition->accelerator);
  }
}

inline static void SbsBaseLayer_update(SbsBaseLayer * layer, SbsBaseLayer * spike_layer) __attribute__((always_inline));
inline static void SbsBaseLayer_update(SbsBaseLayer * layer, SbsBaseLayer * spike_layer)
{
  ASSERT (layer != NULL);
  ASSERT (spike_layer != NULL);
  if ((layer != NULL) && (spike_layer != NULL))
  {
    int i;
    SbsLayerPartition *  update_partition = NULL;
    uint16_t             update_partition_rows;
    uint16_t             update_partition_row;
    Multivector * spike_layer_spike_matrix = spike_layer->spike_matrix;

#ifdef DEBUG
    SpikeID   spikeID       = 0;
    Weight * weight_vector  = NULL;
    NeuronState* state_vector;
#endif


    uint16_t kernel_stride  = layer->kernel_stride;
    uint16_t kernel_size    = layer->kernel_size;


    uint16_t layer_row;         /* Row index for navigation on the layer */
    uint16_t layer_column;      /* Column index for navigation on the layer */
    uint16_t kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
    uint16_t kernel_row_pos;    /* Kernel row position for navigation on the spike matrix */
    uint16_t kernel_row;        /* Row index for navigation inside kernel */
    uint16_t kernel_column;     /* Column index for navigation inside kernel */

    uint16_t layer_columns = layer->columns;

    Multivector * update_partition_weight_matrix = NULL;
    SbSUpdateAccelerator * update_partition_accelerator = NULL;
    Multivector * update_partition_state_matrix = NULL;

    WeightShift layer_weight_shift = layer->weight_shift;


    kernel_row_pos = 0, layer_row = 0;
    for (i = 0; i < layer->num_partitions; i ++)
    {
      update_partition = layer->partition_array[i];
      ASSERT(update_partition != NULL);

      update_partition_weight_matrix = update_partition->weight_matrix;
      update_partition_accelerator = update_partition->accelerator;
      update_partition_state_matrix = update_partition->state_matrix;
      update_partition_rows = update_partition_state_matrix->dimension_size[0];

      Accelerator_setup (update_partition_accelerator,
                         &update_partition->profile,
                         UPDATE_MODE);

      /* Update begins */
      for (update_partition_row = 0;
           update_partition_row < update_partition_rows;
           update_partition_row ++,
           kernel_row_pos += kernel_stride, layer_row ++)
      {
        for (kernel_column_pos = 0, layer_column = 0;
            layer_column < layer_columns;
             kernel_column_pos += kernel_stride, layer_column ++)
        {
#ifndef DEBUG
/*-------------------------- NOT debugging (FAST MODE) ----------------------*/
          Accelerator_giveStateVector (update_partition_accelerator,
                                       Multivector_2DAccess(update_partition_state_matrix,
                                                            update_partition_row,
                                                            layer_column));

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              if (layer_weight_shift == COLUMN_SHIFT)
              {
                Accelerator_giveWeightVector (update_partition_accelerator,
                                              Multivector_3DAccess (update_partition_weight_matrix,
                                                                    kernel_row,
                                                                    kernel_column,
                                                                    *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix,
                                                                                                      kernel_row_pos + kernel_row,
                                                                                                      kernel_column_pos + kernel_column)));
              }
              else
              {
                Accelerator_giveWeightVector (update_partition_accelerator,
                                              Multivector_3DAccess (update_partition_weight_matrix,
                                                                    kernel_column,
                                                                    kernel_row,
                                                                    *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix,
                                                                                                      kernel_row_pos + kernel_row,
                                                                                                      kernel_column_pos + kernel_column)));
              }
            }
          }
#else
          state_vector = Multivector_2DAccess(update_partition_state_matrix, update_partition_row, layer_column);

          Accelerator_giveStateVector (update_partition_accelerator, state_vector);

          for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
          {
            for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
            {
              spikeID = *(SpikeID *) Multivector_2DAccess(spike_layer_spike_matrix, kernel_row_pos + kernel_row, kernel_column_pos + kernel_column);

              ASSERT(layer->vector_size == update_partition->weight_matrix->dimension_size[3]);

              if (layer_weight_shift == COLUMN_SHIFT)
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_row, kernel_column, spikeID);
              }
              else
              {
                weight_vector = Multivector_3DAccess (update_partition_weight_matrix, kernel_column, kernel_row, spikeID);
              }

              Accelerator_giveWeightVector (update_partition_accelerator, weight_vector);
            }
          }
#endif
        }
      }
      /* Update ends */
      Accelerator_start (update_partition_accelerator);
    }
  }
}

/*****************************************************************************/

static SbsNetwork * SbsBaseNetwork_new(void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc (sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
    memset (network, 0x0, sizeof(SbsBaseNetwork));
    network->vtbl = _SbsNetwork;
    network->input_label = (uint8_t) -1;
    network->inferred_output = (uint8_t) -1;

    sgenrand (666); /*TODO: Create MT19937 object wrapper */
  }

  ASSERT(network->size == 0);
  ASSERT(network->layer_array == NULL);

  return (SbsNetwork *) network;
}

static void SbsBaseNetwork_delete(SbsNetwork ** network_ptr)
{
  ASSERT(network_ptr != NULL);
  ASSERT(*network_ptr != NULL);

  if ((network_ptr != NULL) && (*network_ptr != NULL))
  {
    SbsBaseNetwork ** network = (SbsBaseNetwork **) network_ptr;
    while (0 < (*network)->size)
      SbsBaseLayer_delete((SbsLayer **)&(*network)->layer_array[--((*network)->size)]);

    free(*network);
    *network = NULL;
  }
}

static void SbsBaseNetwork_giveLayer(SbsNetwork * network_ptr, SbsLayer * layer)
{
  ASSERT(network_ptr != NULL);
  ASSERT(layer != NULL);

  if ((network_ptr != NULL) && (layer != NULL))
  {
    SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
    SbsBaseLayer ** layer_array = network->layer_array;
    uint8_t size = network->size;

    ASSERT(size < 0xFF);

    layer_array = realloc(layer_array, (size + 1) * sizeof(SbsBaseLayer *));

    ASSERT(layer_array != NULL);

    if (layer_array != NULL)
    {
        layer_array[size] = (SbsBaseLayer *)layer;

        network->layer_array = layer_array;
        network->size ++;
    }
  }
}

static void SbsBaseNetwork_loadInput(SbsNetwork * network_ptr, char * file_name)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(1 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(*network->layer_array != NULL);
  ASSERT(file_name != NULL);

  if ((network != NULL)
      && (1 <= network->size)
      && (network->layer_array != NULL) && (*network->layer_array != NULL)
      && (file_name != NULL))
  {
    SbsBaseLayer_loadInput (network->layer_array[0], file_name,
                            &network->input_label);
  }
}

#define SBS_LEARNING_THRESHOLD 0.00001

static void SbsBaseLayer_learningDeltaMSE(SbsBaseLayer * layer, SbsBaseLayer * prev_layer)
{
  ASSERT (layer != NULL);
  ASSERT (layer->partition_array != NULL);
  ASSERT (layer->partition_array[0] != NULL);
  ASSERT (layer->partition_array[0]->state_matrix != NULL);
  ASSERT (layer->partition_array[0]->state_matrix->dimensionality == 3);
  ASSERT (layer->partition_array[0]->weight_matrix != NULL);
  ASSERT (layer->partition_array[0]->weight_matrix->dimensionality == 4);


  ASSERT (prev_layer != NULL);
  ASSERT (prev_layer->partition_array != NULL);
  ASSERT (prev_layer->partition_array[0] != NULL);
  ASSERT (prev_layer->partition_array[0]->state_matrix != NULL);
  ASSERT (prev_layer->partition_array[0]->state_matrix->dimensionality == 3);

  if (layer != NULL && prev_layer != NULL)
  {
    Multivector * h_matrix = layer->partition_array[0]->state_matrix;
    int h_neurons = h_matrix->dimension_size[2];

    Multivector * w_matrix = layer->partition_array[0]->weight_matrix;
    int w_rows = w_matrix->dimension_size[0];
    int w_cols = w_matrix->dimension_size[1];
    int w_spikes = w_matrix->dimension_size[2];
    int w_neurons = w_matrix->dimension_size[3];

    Multivector * prev_layer_h_matrix = prev_layer->partition_array[0]->state_matrix;

    int row;
    int col;
    int spike;
    int i;

    Weight w;
    NeuronState h;

    float * reco_vector = layer->learning_data.reco_vector;
    float * delat_vector = layer->learning_data.delat_vector;

    ASSERT (reco_vector != NULL);
    ASSERT (delat_vector != NULL);

    ASSERT (w_neurons == h_neurons);

    // 1)
    for (row = 0; row < w_rows; row++)
    {
      for (col = 0; col < w_cols; col++)
      {
        for (spike = 0; spike < w_spikes; spike++)
        {
          reco_vector[spike] = 0;
          for (i = 0; i < w_neurons; i++)
          {
            w = ((Weight *) Multivector_3DAccess (w_matrix, row, col, spike))[i];
            h = *((NeuronState *) Multivector_3DAccess (h_matrix, 0, 0, i));

            reco_vector[spike] += w * h;
          }
        }
      }
    }

    // 2)
    for (spike = 0; spike < w_spikes; spike++)
    {
      delat_vector[spike] = *(NeuronState*) Multivector_3DAccess (prev_layer_h_matrix, 0, 0, spike) - reco_vector[spike];
    }

    // 3)
    for (spike = 0; spike < w_spikes; spike++)
    {
      for (i = 0; i < h_neurons; i++)
      {
        h = *((NeuronState *) Multivector_3DAccess (h_matrix, 0, 0, i));

        *((double*) Multivector_2DAccess (layer->learning_data.omega_matrix, spike, i)) += ((double)delat_vector[spike]) * ((double)h);
      }
    }

    layer->learning_data.current_pattern ++;

    if (layer->learning_data.current_pattern == layer->learning_data.number_of_patterns)
    {
      double * b_vector = layer->learning_data.b_vector;
      double temp;
      ASSERT (b_vector != NULL);

      layer->learning_data.current_pattern = 0;

      temp = layer->learning_data.gama / layer->learning_data.number_of_patterns;

      for (spike = 0; spike < w_spikes; spike++)
      {
        for (i = 0; i < h_neurons; i++)
        {
          *((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)) =
              ((Weight*) Multivector_3DAccess (w_matrix, 0, 0, spike))[i] +
              temp * (*((double*) Multivector_2DAccess (layer->learning_data.omega_matrix, spike, i)));

          if (*((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)) < 0.0)
          {
            *((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)) = SBS_LEARNING_THRESHOLD;
          }
        }
      }

      memset (b_vector, 0, sizeof(double) * w_neurons);
      for (i = 0; i < w_neurons; i++)
      {
        for (spike = 0; spike < w_spikes; spike++)
        {
          b_vector[i] += *((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i));
        }
      }

      for (i = 0; i < w_neurons; i++)
      {
        temp = 1.0 / b_vector[i];
        for (spike = 0; spike < w_spikes; spike++)
        {
          ((Weight *) Multivector_3DAccess (w_matrix, 0, 0, spike))[i] =
              temp * (*((double *) Multivector_2DAccess (layer->learning_data.b_matrix, spike, i)));

          ASSERT(0.0 <= ((Weight * ) Multivector_3DAccess (w_matrix, 0, 0, spike))[i]
                  && ((Weight * ) Multivector_3DAccess (w_matrix, 0, 0, spike))[i] <= 1.0);
        }
      }
    }
  }
}

static void SbsBaseLayer_learning(SbsBaseLayer * layer, SbsBaseLayer * prev_layer)
{
  ASSERT(layer != NULL);
  if (layer != NULL)
    switch (layer->learning_data.learning_rule)
    {
      case SBS_LEARNING_NONE:
        break;
      case SBS_LEARNING_DELTA_MSE:
        // TODO: MSE in fixed-point
        SbsBaseLayer_learningDeltaMSE(layer, prev_layer);
        break;
      case SBS_LEARNING_RELATIVE_ENTROPY:
        break;
      default:
        ASSERT (NULL);
    }
}

static void SbsBaseNetwork_updateInferredOutput(SbsBaseNetwork * network)
{
  ASSERT (network != NULL);

  if (network != NULL)
  {
    NeuronState max_value = 0;
    SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
    NeuronState * output_state_vector = NULL;
    uint16_t output_vector_size = 0;
    int i;

    SbsBaseLayer_getOutputVector (output_layer, &output_state_vector,
                                  &output_vector_size);

    ASSERT (output_state_vector != NULL);
    ASSERT (0 < output_vector_size);

    for (i = 0; i < output_vector_size; i++)
    {
      NeuronState h = output_state_vector[i]; /* Ensure data alignment */
      if (max_value < h)
      {
        network->inferred_output = i;
        max_value = h;
      }
    }
  }
}

static void SbsBaseNetwork_updateCycle(SbsNetwork * network_ptr, uint16_t cycles)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  SbsBaseLayer * input_layer = NULL;

  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    int i;

    /* Initialize all layers except the input-layer */
    for (i = 0; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize(network->layer_array[i]);
      SbsBaseLayer_cacheFlush(network->layer_array[i]);
    }

    input_layer = network->layer_array[0];

    /************************ Begins Update cycle ****************************/
    while (cycles--)
    {
      SbsBaseLayer_generateSpikes (input_layer);

      for (i = 1; i <= network->size - 1; i++)
      {
        SbsBaseLayer_update (network->layer_array[i],
                             network->layer_array[i - 1]);
      }
    }
    /************************ Ends Update cycle ******************************/

    /************************ Begins Learning cycle **************************/
    for (i = 1; i <= network->size - 1; i++)
    {
      SbsBaseLayer_learning (network->layer_array[i],
                             network->layer_array[i - 1]);
    }
    /************************ Ends Learning cycle ****************************/

    /************************ Ends Update cycle ****************************/

    /************************ Get inferred output **************************/
    SbsBaseNetwork_updateInferredOutput (network);
  }
}

static uint8_t SbsBaseNetwork_getInferredOutput(SbsNetwork * network)
{
  uint8_t inferred_output = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    inferred_output = ((SbsBaseNetwork *) network)->inferred_output;
  }

  return inferred_output;
}

static uint8_t SbsBaseNetwork_getInputLabel(SbsNetwork * network)
{
  uint8_t input_label = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    input_label = ((SbsBaseNetwork *) network)->input_label;
  }

  return input_label;
}

static void SbsBaseNetwork_getOutputVector(SbsNetwork * network_ptr,
                                           NeuronState ** output_vector,
                                           uint16_t * output_vector_size)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(0 < network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(network->layer_array[network->size - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != NULL);

  if ((network != NULL)
      && (0 < network->size)
      && (network->layer_array != NULL)
      && (network->layer_array[network->size - 1] != NULL)
      && (output_vector != NULL)
      && (output_vector_size != NULL))
  {
    SbsBaseLayer_getOutputVector (network->layer_array[network->size - 1],
                                  output_vector, output_vector_size);
  }
}

static void SbsBaseNetwork_printStatistics (SbsNetwork * network)
{

}
/*****************************************************************************/

static SbsLayer * SbsInputLayer_new(SbsLayerType layer_type,
                                    uint16_t rows,
                                    uint16_t columns,
                                    uint16_t neurons)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        0,
                                        0, ROW_SHIFT);
}

static SbsLayer * SbsConvolutionLayer_new(SbsLayerType layer_type,
                                          uint16_t rows,
                                          uint16_t columns,
                                          uint16_t neurons,
                                          uint16_t kernel_size,
                                          WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        kernel_size,
                                        1,
                                        weight_shift);
}

static SbsLayer * SbsPoolingLayer_new(SbsLayerType layer_type,
                                      uint16_t rows,
                                      uint16_t columns,
                                      uint16_t neurons,
                                      uint16_t kernel_size,
                                      WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        rows,
                                        columns,
                                        neurons,
                                        kernel_size,
                                        kernel_size,
                                        weight_shift);
}

static SbsLayer * SbsFullyConnectedLayer_new(SbsLayerType layer_type,
                                             uint16_t neurons,
                                             uint16_t kernel_size,
                                             WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        1,
                                        1,
                                        neurons,
                                        kernel_size,
                                        1,
                                        weight_shift);
}

static SbsLayer * SbsOutputLayer_new(SbsLayerType layer_type,
                                     uint16_t neurons,
                                     WeightShift weight_shift)
{
  return (SbsLayer *) SbsBaseLayer_new (layer_type,
                                        1,
                                        1,
                                        neurons,
                                        1,
                                        1,
                                        weight_shift);
}
/*****************************************************************************/

static SbsWeightMatrix SbsWeightMatrix_new (uint16_t rows,
                                            uint16_t columns,
                                            uint16_t depth,
                                            uint16_t neurons,
                                            char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new(NULL, sizeof(float), 4, rows, columns, depth, neurons);

    ASSERT(weight_watrix != NULL);
    ASSERT(weight_watrix->dimensionality == 4);
    ASSERT(weight_watrix->data != NULL);
    ASSERT(weight_watrix->dimension_size[0] == rows);
    ASSERT(weight_watrix->dimension_size[1] == columns);
    ASSERT(weight_watrix->dimension_size[2] == depth);
    ASSERT(weight_watrix->dimension_size[3] == neurons);

    if ((weight_watrix != NULL)
        && (weight_watrix->dimensionality == 4)
        && (weight_watrix->data != NULL)
        && (weight_watrix->dimension_size[0] == rows)
        && (weight_watrix->dimension_size[1] == columns)
        && (weight_watrix->dimension_size[2] == depth)
        && (weight_watrix->dimension_size[3] == neurons))
    {
      FIL fil; /* File object */
      FRESULT rc;
      rc = f_open (&fil, file_name, FA_READ);
      ASSERT(rc == FR_OK);

      if (rc == FR_OK)
      {
        Multivector * weight_watrix_16bit = NULL;
        size_t read_size;
        size_t data_size = rows * columns * depth * neurons * sizeof(float); // TODO: Define the data type of the file
        rc = f_read (&fil, weight_watrix->data, data_size, &read_size);
        ASSERT((rc == FR_OK) && (read_size == data_size));
        f_close (&fil);

        weight_watrix_16bit = Multivector_convert2fixed16(weight_watrix->memory_def_parent, weight_watrix);

        ASSERT(weight_watrix_16bit != NULL);

        Multivector_delete (&weight_watrix);

        weight_watrix = weight_watrix_16bit;
      }
      else Multivector_delete (&weight_watrix);
    }
  }

  return weight_watrix;
}

/*****************************************************************************/

SbsNetwork _SbsNetwork = {SbsBaseNetwork_new,
                          SbsBaseNetwork_delete,
                          SbsBaseNetwork_giveLayer,
                          SbsBaseNetwork_loadInput,
                          SbsBaseNetwork_updateCycle,
                          SbsBaseNetwork_getInferredOutput,
                          SbsBaseNetwork_getInputLabel,
                          SbsBaseNetwork_getOutputVector,
                          SbsBaseNetwork_printStatistics};

SbsLayer _SbsLayer = {SbsBaseLayer_new,
                      SbsBaseLayer_delete,
                      SbsBaseLayer_setEpsilon,
                      SbsBaseLayer_setLearningRule,
                      SbsBaseLayer_giveWeights};

SbsNew sbs_new = {SbsBaseNetwork_new,
                  SbsBaseLayer_new,
                  SbsWeightMatrix_new,
                  SbsInputLayer_new,
                  SbsConvolutionLayer_new,
                  SbsPoolingLayer_new,
                  SbsFullyConnectedLayer_new,
                  SbsOutputLayer_new};


/*****************************************************************************/
