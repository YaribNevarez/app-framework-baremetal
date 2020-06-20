#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

#include <stdio.h>
#include <string.h>

#define MT19937_HW false

#if MT19937_HW
typedef unsigned int MT19937;

static unsigned int MT19937_flags_ = 0;

typedef enum
{
  INITIALIZED = 1 << 0
} MT19937Flags;

static unsigned int MT19937_initialized (unsigned int instance)
{
  return MT19937_flags_ & INITIALIZED;
}

/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */

/* Tempering parameters */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned int mt[N]; /* the array for the state vector  */
static unsigned int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializing the array with a NONZERO seed */
static void MT19937_sgenrand (unsigned int instance, unsigned int seed)
{
  /* setting initial seeds to mt[N] using         */
  /* the generator Line 25 of Table 1 in          */
  /* [KNUTH 1981, The Art of Computer Programming */
  /*    Vol. 2 (2nd Ed.), pp102]                  */
  mt[0] = seed & 0xffffffff;
  for (mti = 1; mti < N; mti++)
  {
#pragma HLS pipeline
    mt[mti] = (69069 * mt[mti - 1]) & 0xffffffff;
  }

  MT19937_flags_ |= INITIALIZED;
}

static unsigned int MT19937_rand (unsigned int instance)
{
#pragma HLS inline off
  unsigned int y;
  static unsigned int mag01[2] = { 0x0, MATRIX_A };
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

  if (mti >= N)
  { /* generate N words at one time */
    int kk;

    if (mti == N + 1)
    {/* if sgenrand() has not been called, */
      MT19937_sgenrand (instance, 4357); /* a default initial seed is used   */
    }

    for (kk = 0; kk < N - M; kk++)
    {
#pragma HLS pipeline
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
    }

    for (; kk < N - 1; kk++)
    {
#pragma HLS pipeline
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
    }

    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

    mti = 0;
  }

  y = mt[mti++];
  y ^= TEMPERING_SHIFT_U(y);
  y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
  y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
  y ^= TEMPERING_SHIFT_L(y);

  return y;
}
#endif


typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;


#define CHANNEL_WIDTH         32
#define STATE_VECTOR_WIDTH    16
#define SPIKE_VECTOR_WIDTH    16

#define MAX_VECTOR_SIZE   50

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0xFFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (  0xFF & (((unsigned int)(d)) >> 19))

#define DATA_SIZE     sizeof(short) // Bytes
#define BUFFER_SIZE   ((MAX_VECTOR_SIZE + 1) * DATA_SIZE) / sizeof(Data32) + 1

void sbs_spike_50 (ap_uint<CHANNEL_WIDTH> * frame_in,
                   ap_uint<CHANNEL_WIDTH> * spike_out,
                   int * debug,
                   int layerSize,
                   int vectorSize)
{
#pragma HLS INTERFACE m_axi     port=frame_in    offset=slave   bundle=DATA_BUS
#pragma HLS INTERFACE m_axi     port=spike_out   offset=slave   bundle=DATA_BUS

#pragma HLS INTERFACE s_axilite port=frame_in    bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=spike_out   bundle=CRTL_BUS

#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static ap_uint<CHANNEL_WIDTH> input_vector[MAX_VECTOR_SIZE / (CHANNEL_WIDTH / STATE_VECTOR_WIDTH)];
#pragma HLS array_partition variable=input_vector cyclic factor=1 dim=1

  static float data[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=data complete

  unsigned char index_spike;
  unsigned int debug_flags;

  static float random_value;

  static ap_uint<CHANNEL_WIDTH> input;
  static Data32 register_B;

  static float sum;

  unsigned int frame_index = 0;

  unsigned int spike_index = 0;
  unsigned int spike_data = 0;

#if MT19937_HW
  if (!MT19937_initialized (0))
  {
    MT19937_sgenrand (0, 666);
  }
#endif

  for (int ip_index = 0; ip_index < layerSize; ip_index++)
  {
#if MT19937_HW
    random_value = ((float) MT19937_rand (0)) / ((float) 0xFFFFFFFF);
#else
#pragma HLS pipeline
    register_B.u32 = frame_in[frame_index++];

    random_value = register_B.f32;
#endif

    memcpy (input_vector, &frame_in[frame_index], (vectorSize >> 1) * sizeof(ap_uint<CHANNEL_WIDTH> ));
    frame_index += vectorSize >> 1;
    for (int index = 0; index < vectorSize; index += CHANNEL_WIDTH / STATE_VECTOR_WIDTH)
    {
#pragma HLS pipeline
      input = input_vector[index >> 1];

      register_B.u32 = DATA16_TO_FLOAT32(input >> STATE_VECTOR_WIDTH * 0);
      data[index] = register_B.f32;

      if (index + 1 < vectorSize)
      {
#pragma HLS pipeline
        register_B.u32 = DATA16_TO_FLOAT32(input >> STATE_VECTOR_WIDTH * 1);
        data[index + 1] = register_B.f32;
      }
    }

    sum = 0.0f;
    for (short spikeID = 0; spikeID < vectorSize; spikeID++)
    {
#pragma HLS pipeline
      if (sum < random_value)
      {
#pragma HLS pipeline
        sum += data[spikeID];
        if ((random_value <= sum) || (spikeID == vectorSize - 1))
        {
#pragma HLS pipeline
          if (ip_index % 2 == 0)
          {
#pragma HLS pipeline
            spike_data = spikeID;
          }
          else
          {
#pragma HLS pipeline
            spike_data |= ((ap_uint<CHANNEL_WIDTH>)spikeID) << SPIKE_VECTOR_WIDTH;
          }

          if ((ip_index % 2 == 1) || (ip_index == layerSize - 1))
          {
#pragma HLS pipeline
            spike_out[spike_index ++] = spike_data;
          }
        }
      }
    }
  }
}
