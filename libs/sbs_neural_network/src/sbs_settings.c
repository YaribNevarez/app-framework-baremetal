/*
 * sbs_settings.c
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */


//#define DEBUG //  -g3 -O0 -DDEBUG

#include "sbs_settings.h"

Format state_matrix_format;
Format weight_matrix_format;
Format spike_matrix_format;

SbsSettings SbsSettings_ =
#ifndef STANDARD_FLOATINGPOINT
{
    .state_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(uint16_t),
        .mantissa_bitlength = 11
    },
    .weight_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(uint8_t),
        .mantissa_bitlength = 4
    },
    .spike_matrix_format =
    {
        .representation = FIXED_POINT,
        .size = sizeof(uint16_t),
        .mantissa_bitlength = 0
    },
    .learning_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(double),
        .mantissa_bitlength = 0
    },
    .weight_matrix_format_file_system =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    },
    .input_matrix_format_file_system =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    }
};
#else
{
    .state_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    },
    .weight_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    },
    .spike_matrix_format =
    {
        .representation = FIXED_POINT,
        .size = sizeof(uint32_t),
        .mantissa_bitlength = 0
    },
    .learning_matrix_format =
    {
        .representation = FLOAT,
        .size = sizeof(double),
        .mantissa_bitlength = 0
    },
    .weight_matrix_format_file_system =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    },
    .input_matrix_format_file_system =
    {
        .representation = FLOAT,
        .size = sizeof(float),
        .mantissa_bitlength = 0
    }
};
#endif
