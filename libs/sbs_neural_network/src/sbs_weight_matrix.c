/*
 * sbs_weight_matrix.c
 *
 *  Created on: Aug 1, 2020
 *      Author: Yarib Nevarez
 */


//#define DEBUG //  -g3 -O0 -DDEBUG

#include "sbs_weight_matrix.h"
#include "sbs_settings.h"
#include "multivector.h"

#include "ff.h"
#include "miscellaneous.h"
#include "string.h"


/*****************************************************************************/

SbsWeightMatrix SbsWeightMatrix_new (uint16_t rows,
                                     uint16_t columns,
                                     uint16_t depth,
                                     uint16_t neurons,
                                     size_t memory_padding,
                                     char * file_name)
{
  Multivector * weight_watrix = NULL;

  ASSERT(file_name != NULL);

  if (file_name != NULL)
  {
    weight_watrix = Multivector_new(NULL,
                                    &SbsSettings_.weight_matrix_format_file_system,
                                    memory_padding,
                                    4,
                                    rows,
                                    columns,
                                    depth,
                                    neurons);

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
        Multivector * weight_watrix_internal = NULL;
        size_t read_size;
        size_t data_size = rows * columns * depth * neurons * SbsSettings_.weight_matrix_format_file_system.size; // TODO: Define the data type of the file
        rc = f_read (&fil, weight_watrix->data, data_size, &read_size);
        ASSERT((rc == FR_OK) && (read_size == data_size));
        f_close (&fil);

        ///////////////////////////////////////////////////////////////////////
        Histogram histogram;
        Multivector_getHistogram (weight_watrix, &histogram);
        if (histogram.bin_array_len < 32)
        {
          for (int i = 0; i < histogram.bin_array_len; i ++)
          {
            printf ("'-%d':%.2f, ", i, 100.0 * ((float) histogram.bin_array[i]) / ((float) histogram.total_samples));
          }

          printf ("\n");
        }
        ///////////////////////////////////////////////////////////////////////

        if (memcmp(&weight_watrix->format,
                   &SbsSettings_.weight_matrix_format,
                   sizeof(SbsSettings_.weight_matrix_format)) != 0)
        {

          weight_watrix_internal = Multivector_reformat (weight_watrix->memory_def_parent,
                                                         weight_watrix,
                                                         &SbsSettings_.weight_matrix_format,
                                                         memory_padding);

          ASSERT(weight_watrix_internal != NULL);

          Multivector_delete (&weight_watrix);

          weight_watrix = weight_watrix_internal;
        }
      }
      else Multivector_delete (&weight_watrix);
    }
  }

  return weight_watrix;
}
