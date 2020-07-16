/*
 * multivector.c
 *
 *  Created on: Feb 24th, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "multivector.h"
#include "miscellaneous.h"

#include "stdio.h"
#include "stdarg.h"
#include "stdlib.h"
#include "string.h"

#include "xil_cache.h"

/***************** Macros (Inline Functions) Definitions *********************/

//#define MULTIVECTOR_USE_POINTER_ARITHMETICS

/**************************** Type Definitions *******************************/

typedef double   MDouble_1024_10[1024][10];
typedef uint32_t M32Bit_24_24[24][24];
typedef uint16_t M16Bit_24_24[24][24];
typedef uint32_t M32Bit_12_24[12][24];
typedef uint32_t M32Bit_24_24_50[24][24][50];
typedef uint16_t M16Bit_24_24_50[24][24][50];
typedef uint32_t M32Bit_12_24_32[12][24][32];
typedef uint32_t M32Bit_24_24_32[24][24][32];
typedef uint16_t M16Bit_24_24_32[24][24][32];
typedef uint32_t M32Bit_1_1_50_32[1][1][50][32];
typedef uint16_t   M16Bit_1_1_50_32[1][1][50][32];
typedef uint8_t   M8Bit_1_1_50_32[1][1][50][32];
typedef uint32_t M32Bit_6_12_32[6][12][32];
typedef uint32_t M32Bit_12_12_32[12][12][32];
typedef uint16_t M16Bit_12_12_32[12][12][32];
typedef uint32_t M32Bit_12_12[12][12];
typedef uint16_t M16Bit_12_12[12][12];
typedef uint32_t M32Bit_6_12[6][12];
typedef uint32_t M32Bit_2_2_32_32[2][2][32][32];
typedef uint16_t   M16Bit_2_2_32_32[2][2][32][32];
typedef uint8_t   M8Bit_2_2_32_32[2][2][32][32];
typedef uint32_t M32Bit_8_8_64[8][8][64];
typedef uint16_t M16Bit_8_8_64[8][8][64];
typedef uint32_t M32Bit_4_8_64[4][8][64];
typedef uint16_t M16Bit_4_8_64[4][8][64];
typedef uint32_t M32Bit_8_8[8][8];
typedef uint16_t M16Bit_8_8[8][8];
typedef uint32_t M32Bit_4_8[8][8];
typedef uint32_t M32Bit_5_5_32_64[5][5][32][64];
typedef uint16_t   M16Bit_5_5_32_64[5][5][32][64];
typedef uint8_t   M8Bit_5_5_32_64[5][5][32][64];
typedef uint32_t M32Bit_2_4_64[2][4][64];
typedef uint32_t M32Bit_4_4_64[4][4][64];
typedef uint16_t M16Bit_4_4_64[4][4][64];
typedef uint32_t M32Bit_4_4[4][4];
typedef uint16_t M16Bit_4_4[4][4];
typedef uint32_t M32Bit_2_4[2][4];
typedef uint32_t M32Bit_2_2_64_64[2][2][64][64];
typedef uint16_t   M16Bit_2_2_64_64[2][2][64][64];
typedef uint8_t   M8Bit_2_2_64_64[2][2][64][64];
typedef uint32_t M32Bit_1_1_1024[1][1][1024];
typedef uint16_t M16Bit_1_1_1024[1][1][1024];
typedef uint32_t M32Bit_1_1[1][1];
typedef uint16_t M16Bit_1_1[1][1];
typedef uint32_t M32Bit_4_4_64_1024[4][4][64][1024];
typedef uint16_t   M16Bit_4_4_64_1024[4][4][64][1024];
typedef uint8_t   M8Bit_4_4_64_1024[4][4][64][1024];
typedef uint32_t M32Bit_1_1_10[1][1][10];
typedef uint16_t M16Bit_1_1_10[1][1][10];
typedef uint32_t M32Bit_1_1_1024_10[1][1][1024][10];
typedef uint16_t   M16Bit_1_1_1024_10[1][1][1024][10];
typedef uint8_t   M8Bit_1_1_1024_10[1][1][1024][10];

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
        .type_id = M16BIT_24_24_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_24_24_50_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_24_24_32_ID,
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 3,
        .dimension_size = {24, 24, 32, 0}
    },
    {
        .type_id = M16BIT_24_24_32_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M8BIT_1_1_50_32_ID,
        .data_type_size = sizeof(uint8_t),
        .dimensionality = 4,
        .dimension_size = {1, 1, 50, 32}
    },
    {
        .type_id = M16BIT_1_1_50_32_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_12_12_32_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_12_12_ID,
        .data_type_size = sizeof(uint16_t),
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
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 4,
        .dimension_size = {2, 2, 32, 32}
    },
    {
        .type_id = M8BIT_2_2_32_32_ID,
        .data_type_size = sizeof(uint8_t),
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
        .type_id = M16BIT_8_8_64_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_4_8_64_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_8_8_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_4_8_ID,
        .data_type_size = sizeof(uint16_t),
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
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 4,
        .dimension_size = {5, 5, 32, 64}
    },
    {
        .type_id = M8BIT_5_5_32_64_ID,
        .data_type_size = sizeof(uint8_t),
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
        .type_id = M16BIT_4_4_64_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_4_4_ID,
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 2,
        .dimension_size = {4, 4, 0, 0}
    },
    {
        .type_id = M16BIT_4_4_ID,
        .data_type_size = sizeof(uint16_t),
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
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 4,
        .dimension_size = {2, 2, 64, 64}
    },
    {
        .type_id = M8BIT_2_2_64_64_ID,
        .data_type_size = sizeof(uint8_t),
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
        .type_id = M16BIT_1_1_1024_ID,
        .data_type_size = sizeof(uint16_t),
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
        .type_id = M16BIT_1_1_ID,
        .data_type_size = sizeof(uint16_t),
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
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 4,
        .dimension_size = {4, 4, 64, 1024}
    },
    {
        .type_id = M8BIT_4_4_64_1024_ID,
        .data_type_size = sizeof(uint8_t),
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
        .type_id = M16BIT_1_1_10_ID,
        .data_type_size = sizeof(uint16_t),
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
        .data_type_size = sizeof(uint16_t),
        .dimensionality = 4,
        .dimension_size = {1, 1, 1024, 10}
    },
    {
        .type_id = M8BIT_1_1_1024_10_ID,
        .data_type_size = sizeof(uint8_t),
        .dimensionality = 4,
        .dimension_size = {1, 1, 1024, 10}
    }
};

const unsigned M32BitFormat_list_length = (sizeof(M32BitFormat_list)
                                         / sizeof (M32BitFormat));

MatrixTypeID M32BitFormat_getTypeID (uint8_t data_type_size,
                                     uint8_t dimensionality,
                                     uint16_t * dimension_size)
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

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

Multivector * Multivector_new (MemoryBlock * memory_def,
                               Format * format,
                               size_t memory_padding,
                               uint8_t dimensionality,
                               ...)
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
      size_t alignment_padding;
      va_list argument_list;

      memset (multivector, 0x00, memory_size);

      va_start(argument_list, dimensionality);

      for (data_size = 1, arg = 0; arg < dimensionality; arg ++)
        data_size *= (multivector->dimension_size[arg] = (uint16_t) va_arg(argument_list, int));

      va_end(argument_list);

      data_size *= format->size;

      multivector->memory_def_parent = memory_def;

      multivector->memory_padding = memory_padding;

      // Ensure word alignment
      alignment_padding = data_size % memory_padding;
      if (alignment_padding)
      {
        data_size += memory_padding - alignment_padding;
      }
      multivector->data_size = data_size;

      if (memory_def != NULL)
        multivector->data = MemoryBlock_alloc (memory_def, data_size);
      else
        multivector->data = malloc (data_size);

      ASSERT(multivector->data != NULL);

      if (multivector->data != NULL)
        memset(multivector->data, 0x00, data_size);

      multivector->dimensionality = dimensionality;
      multivector->format = *format;
#ifndef MULTIVECTOR_USE_POINTER_ARITHMETICS
      multivector->type_id = M32BitFormat_getTypeID(format->size,
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

void inline * Multivector_2DAccess (Multivector * multivector,
                                    uint16_t row,
                                    uint16_t column)
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
    size_t data_size = multivector->format.size;

    while (dimensionality-- > 2)
    {
      data_size *= multivector->dimension_size[dimensionality];
    }

    data = multivector->data
        + (row * multivector->dimension_size[1] + column) * data_size;
  }

  return data;
}

void inline * Multivector_3DAccess (Multivector * multivector,
                                    uint16_t row,
                                    uint16_t column,
                                    uint16_t position)
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
    size_t data_size = multivector->format.size;

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
    case M16BIT_24_24_ID:
      return &(*(M16Bit_24_24*) multivector->data)[row][column];
    case M32BIT_12_24_ID:
      return &(*(M32Bit_12_24*) multivector->data)[row][column];
    case M32BIT_24_24_50_ID:
      return &(*(M32Bit_24_24_50*) multivector->data)[row][column];
    case M16BIT_24_24_50_ID:
      return &(*(M16Bit_24_24_50*) multivector->data)[row][column];
    case M32BIT_12_24_32_ID:
      return &(*(M32Bit_12_24_32*) multivector->data)[row][column];
    case M32BIT_24_24_32_ID:
      return &(*(M32Bit_24_24_32*) multivector->data)[row][column];
    case M16BIT_24_24_32_ID:
      return &(*(M16Bit_24_24_32*) multivector->data)[row][column];
    case M32BIT_1_1_50_32_ID:
      return &(*(M32Bit_1_1_50_32*) multivector->data)[row][column];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column];
    case M32BIT_12_12_32_ID:
      return &(*(M32Bit_12_12_32*) multivector->data)[row][column];
    case M16BIT_12_12_32_ID:
      return &(*(M16Bit_12_12_32*) multivector->data)[row][column];
    case M32BIT_12_12_ID:
      return &(*(M32Bit_12_12*) multivector->data)[row][column];
    case M16BIT_12_12_ID:
      return &(*(M16Bit_12_12*) multivector->data)[row][column];
    case M32BIT_6_12_ID:
      return &(*(M32Bit_6_12*) multivector->data)[row][column];
    case M32BIT_2_2_32_32_ID:
      return &(*(M32Bit_2_2_32_32*) multivector->data)[row][column];
    case M32BIT_8_8_64_ID:
      return &(*(M32Bit_8_8_64*) multivector->data)[row][column];
    case M16BIT_8_8_64_ID:
      return &(*(M16Bit_8_8_64*) multivector->data)[row][column];
    case M32BIT_4_8_64_ID:
      return &(*(M32Bit_4_8_64*) multivector->data)[row][column];
    case M16BIT_4_8_64_ID:
      return &(*(M16Bit_4_8_64*) multivector->data)[row][column];
    case M32BIT_8_8_ID:
      return &(*(M32Bit_8_8*) multivector->data)[row][column];
    case M16BIT_8_8_ID:
      return &(*(M16Bit_8_8*) multivector->data)[row][column];
    case M32BIT_5_5_32_64_ID:
      return &(*(M32Bit_5_5_32_64*) multivector->data)[row][column];
    case M32BIT_2_4_64_ID:
      return &(*(M32Bit_2_4_64*) multivector->data)[row][column];
    case M32BIT_4_4_64_ID:
      return &(*(M32Bit_4_4_64*) multivector->data)[row][column];
    case M16BIT_4_4_64_ID:
      return &(*(M16Bit_4_4_64*) multivector->data)[row][column];
    case M32BIT_4_4_ID:
      return &(*(M32Bit_4_4*) multivector->data)[row][column];
    case M16BIT_4_4_ID:
      return &(*(M16Bit_4_4*) multivector->data)[row][column];
    case M32BIT_2_2_64_64_ID:
      return &(*(M32Bit_2_2_64_64*) multivector->data)[row][column];
    case M32BIT_1_1_1024_ID:
      return &(*(M32Bit_1_1_1024*) multivector->data)[row][column];
    case M16BIT_1_1_1024_ID:
      return &(*(M16Bit_1_1_1024*) multivector->data)[row][column];
    case M32BIT_1_1_ID:
      return &(*(M32Bit_1_1*) multivector->data)[row][column];
    case M16BIT_1_1_ID:
      return &(*(M16Bit_1_1*) multivector->data)[row][column];
    case M32BIT_4_4_64_1024_ID:
      return &(*(M32Bit_4_4_64_1024*) multivector->data)[row][column];
    case M32BIT_1_1_10_ID:
      return &(*(M32Bit_1_1_10*) multivector->data)[row][column];
    case M16BIT_1_1_10_ID:
      return &(*(M16Bit_1_1_10*) multivector->data)[row][column];
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
    case M8BIT_1_1_50_32_ID:
        return &(*(M8Bit_1_1_50_32*) multivector->data)[row][column][position];
    case M32BIT_6_12_32_ID:
      return &(*(M32Bit_6_12_32*) multivector->data)[row][column][position];
    case M32BIT_2_2_32_32_ID:
      return &(*(M32Bit_2_2_32_32*) multivector->data)[row][column][position];
    case M16BIT_2_2_32_32_ID:
        return &(*(M16Bit_2_2_32_32*) multivector->data)[row][column][position];
    case M8BIT_2_2_32_32_ID:
        return &(*(M8Bit_2_2_32_32*) multivector->data)[row][column][position];
    case M32BIT_8_8_64_ID:
      return &(*(M32Bit_8_8_64*) multivector->data)[row][column][position];
    case M32BIT_5_5_32_64_ID:
      return &(*(M32Bit_5_5_32_64*) multivector->data)[row][column][position];
    case M16BIT_5_5_32_64_ID:
      return &(*(M16Bit_5_5_32_64*) multivector->data)[row][column][position];
    case M8BIT_5_5_32_64_ID:
      return &(*(M8Bit_5_5_32_64*) multivector->data)[row][column][position];
    case M32BIT_2_4_64_ID:
      return &(*(M32Bit_2_4_64*) multivector->data)[row][column][position];
    case M32BIT_2_2_64_64_ID:
      return &(*(M32Bit_2_2_64_64*) multivector->data)[row][column][position];
    case M16BIT_2_2_64_64_ID:
      return &(*(M16Bit_2_2_64_64*) multivector->data)[row][column][position];
    case M8BIT_2_2_64_64_ID:
      return &(*(M8Bit_2_2_64_64*) multivector->data)[row][column][position];
    case M32BIT_1_1_1024_ID:
      return &(*(M32Bit_1_1_1024*) multivector->data)[row][column][position];
    case M16BIT_1_1_1024_ID:
      return &(*(M16Bit_1_1_1024*) multivector->data)[row][column][position];
    case M32BIT_4_4_64_1024_ID:
      return &(*(M32Bit_4_4_64_1024*) multivector->data)[row][column][position];
    case M16BIT_4_4_64_1024_ID:
      return &(*(M16Bit_4_4_64_1024*) multivector->data)[row][column][position];
    case M8BIT_4_4_64_1024_ID:
      return &(*(M8Bit_4_4_64_1024*) multivector->data)[row][column][position];
    case M32BIT_1_1_10_ID:
      return &(*(M32Bit_1_1_10*) multivector->data)[row][column][position];
    case M16BIT_1_1_10_ID:
      return &(*(M16Bit_1_1_10*) multivector->data)[row][column][position];
    case M32BIT_1_1_1024_10_ID:
      return &(*(M32Bit_1_1_1024_10*) multivector->data)[row][column][position];
    case M16BIT_1_1_1024_10_ID:
      return &(*(M16Bit_1_1_1024_10*) multivector->data)[row][column][position];
    case M8BIT_1_1_1024_10_ID:
      return &(*(M8Bit_1_1_1024_10*) multivector->data)[row][column][position];
    default:
      ASSERT(0)
      ;
  }
  return NULL;
}
#endif

Multivector * Multivector_duplicate (MemoryBlock * memory_def,
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
      memcpy (duplicate, original, memory_size);

      duplicate->memory_def_parent = memory_def;

      if (memory_def != NULL)
        duplicate->data = MemoryBlock_alloc (memory_def, original->data_size);
      else
        duplicate->data = malloc (original->data_size);

      ASSERT(duplicate->data != NULL);

      if (duplicate->data != NULL)
        memcpy (duplicate->data, original->data, original->data_size);
      else
        return NULL;
    }
  }
  return duplicate;
}

size_t Multivector_dataSize (Multivector * multivector)
{
  size_t data_size = 0;
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    data_size = multivector->data_size;
  }
  return data_size;
}

void Multivector_cacheFlush (Multivector * multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    Xil_DCacheFlushRange ((UINTPTR) multivector->data, multivector->data_size);
  }
}

void Multivector_cacheInvalidate (Multivector * multivector)
{
  ASSERT(multivector != NULL);
  ASSERT(0 < multivector->dimensionality);

  if ((multivector != NULL) && (0 < multivector->dimensionality))
  {
    Xil_DCacheInvalidateRange ((UINTPTR) multivector->data, multivector->data_size);
  }
}

void Multivector_delete (Multivector ** multivector)
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

//void Multivector_float2Fixed (Multivector * multivector, Format * new_format)
//{
//  ASSERT(multivector != NULL);
//
//  if (multivector != NULL)
//  {
//    uint32_t  full_scale = (1 << new_format->mantissa_bitlength) - 1;
//    int       size = Multivector_dataSize(multivector) / multivector->format.size;
//    void *    old_ptr = multivector->data;
//    void *    new_ptr = multivector->data;
//
//    for (int i = 0; i < size; i ++)
//    {
//      switch (multivector->format.size)
//      {
//        case sizeof(float):
//          ((uint32_t*)new_ptr)[i] = (uint32_t) (((float *)old_ptr)[i] * full_scale);
//          break;
//        default:
//          ASSERT(0);
//      }
//    }
//
//    multivector->format = *new_format;
//  }
//}

Multivector * Multivector_reformat (MemoryBlock * memory_def,
                                    Multivector * original,
                                    Format * new_format,
                                    size_t memory_padding)
{
  Multivector * duplicate = NULL;
  ASSERT(original != NULL);
  ASSERT(0 < original->dimensionality);
  ASSERT(new_format != NULL);

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
      size_t data_size;
      size_t alignment_padding;
      int i;

      memcpy (duplicate, original, memory_size);

      duplicate->format = *new_format;

      duplicate->memory_padding = memory_padding;

      for (i = 0; i < original->dimensionality; i++)
        matrix_size *= original->dimension_size[i];

      data_size = new_format->size * matrix_size;

      // Ensure word alignment
      alignment_padding = data_size % memory_padding;
      if (alignment_padding)
      {
        data_size += memory_padding - alignment_padding;
      }

      duplicate->memory_def_parent = memory_def;

      if (memory_def != NULL)
        duplicate->data = MemoryBlock_alloc (memory_def, data_size);
      else
        duplicate->data = malloc (data_size);

      ASSERT(duplicate->data != NULL);

      if (duplicate->data != NULL)
      {
        void * original_data  = original->data;
        void * duplicate_data = duplicate->data;
        int mantissa_index;
        uint32_t data;

        duplicate->data_size = data_size;

        switch (original->format.size)
        {
          case sizeof(float):
                mantissa_index = 23 - new_format->mantissa_bitlength;
            break;
          case sizeof(double):
                mantissa_index = 52 - new_format->mantissa_bitlength;
            break;
          default:
            ASSERT (0);
        }

        for (int i = 0; i < matrix_size; i ++)
        {

          switch (original->format.size)
          {
            case sizeof(float):
                data = (((uint32_t*) original_data)[i] >> mantissa_index);
              break;
            case sizeof(double):
                data = (((uint64_t*) original_data)[i] >> mantissa_index);
              break;
            default:
              ASSERT (0);
          }

          switch (new_format->size)
          {
            case sizeof(uint8_t):
                ((uint8_t *) duplicate_data)[i] = (uint8_t) (0xFF & data);
              break;
            case sizeof(uint16_t):
                ((uint16_t *) duplicate_data)[i] = (uint16_t) (0xFFFF & data);
              break;
            default:
              ASSERT (0);
          }
        }

#ifndef MULTIVECTOR_USE_POINTER_ARITHMETICS
        duplicate->type_id = M32BitFormat_getTypeID (duplicate->format.size,
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

