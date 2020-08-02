/*
 * custom_float.h
 *
 *  Created on: Aug 1st, 2020
 *      Author: Yarib Nevarez
 */

#ifndef LIBS_C_UTILITIES_INC_CUSTOM_FLOAT_H_
#define LIBS_C_UTILITIES_INC_CUSTOM_FLOAT_H_

#define DATA16_TO_FLOAT32(d)  ((0xFFFF & (d)) ? (0x30000000 | (((unsigned int) (0xFFFF & (d))) << 12)) : 0)
#define DATA08_TO_FLOAT32(d)  ((0x00FF & (d)) ? (0x38000000 | (((unsigned int) (0x00FF & (d))) << 19)) : 0)

#define FLOAT32_TO_DATA16(d)  (((0xF0000000 & (unsigned int) (d)) == 0x30000000) ? (0x0000FFFF & (((unsigned int) (d)) >> 12)) : 0)
#define FLOAT32_TO_DATA08(d)  (((0xF8000000 & (unsigned int) (d)) == 0x38000000) ? (0x000000FF & (((unsigned int) (d)) >> 19)) : 0)

#endif /* LIBS_C_UTILITIES_INC_CUSTOM_FLOAT_H_ */
