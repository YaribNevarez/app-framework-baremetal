/*
 * sbs_settings.h
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */

#ifndef SBS_SETTINGS_H_
#define SBS_SETTINGS_H_
#ifdef __cplusplus
extern "C" {
#endif

#include "multivector.h"

typedef struct
{
  Format state_matrix_format;
  Format weight_matrix_format;
  Format spike_matrix_format;
  Format weight_matrix_format_file_system;
  Format input_matrix_format;
  Format input_matrix_format_file_system;
} SbsSettings;


extern SbsSettings SbsSettings_;

#ifdef __cplusplus
}
#endif
#endif /* SBS_SETTINGS_H_ */
