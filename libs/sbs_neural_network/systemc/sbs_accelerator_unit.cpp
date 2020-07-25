#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

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

#define MAX_VECTOR_SIZE       (1024)
#define MAX_SPIKE_MATRIX_SIZE (60*60)

#define DATA16_TO_FLOAT32(d)  ((0xFFFF & (d)) ? (0x30000000 | (((unsigned int) (0xFFFF & (d))) << 12)) : 0)
#define DATA08_TO_FLOAT32(d)  ((0x00FF & (d)) ? (0x38000000 | (((unsigned int) (0x00FF & (d))) << 19)) : 0)

#define FLOAT32_TO_DATA16(d)  (((0xF0000000 & (unsigned int) (d)) == 0x30000000) ? (0x0000FFFF & (((unsigned int) (d)) >> 12)) : 0)
#define FLOAT32_TO_DATA08(d)  (((0xF8000000 & (unsigned int) (d)) == 0x38000000) ? (0x000000FF & (((unsigned int) (d)) >> 19)) : 0)

#define NEGLECTING_CONSTANT   ((float)1e-20)

#define CHANNEL_WIDTH         32
#define STATE_VECTOR_WIDTH    16
#define WEIGHT_VECTOR_WIDTH   8
#define SPIKE_VECTOR_WIDTH    16

typedef ap_axis<CHANNEL_WIDTH, 2, 5, 6> StreamChannel;

unsigned int sbs_accelerator_unit (hls::stream<StreamChannel> &stream_in,
                                   hls::stream<StreamChannel> &stream_out,
                                   int * debug,
                                   int layerSize,
                                   int kernelSize,
                                   int vectorSize,
                                   float epsilon)
{
//#pragma HLS INTERFACE m_axi     port=debug        offset=slave    bundle=DEBUG

#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=kernelSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=epsilon     bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static StreamChannel channel;

  static unsigned short spike_matrix[MAX_SPIKE_MATRIX_SIZE];
//#pragma HLS array_partition variable=spike_matrix block factor=4

  static float state_vector[MAX_VECTOR_SIZE];
//#pragma HLS array_partition variable=state_vector block factor=246

  static float weight_vector[MAX_VECTOR_SIZE];
//#pragma HLS array_partition variable=weight_vector block factor=246

  static float temp_data[MAX_VECTOR_SIZE];
//#pragma HLS array_partition variable=temp_data block factor=246

  static float epsion_over_sum;
  static float random_value;

  static ap_uint<CHANNEL_WIDTH> input;

  static Data32 register_A;
  static Data32 register_B;

  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  static float sum;

  unsigned int debug_index = 0;

  channel.keep = -1;
  channel.strb = -1;

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

#if 32 <= CHANNEL_WIDTH
    register_B.u32 = stream_in.read ().data;
#else
    register_B.u32 = DATA16_TO_FLOAT32(stream_in.read ().data);
#endif

    random_value = register_B.f32;

#endif

    for (int i = 0; i < vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
    {
#pragma HLS pipeline
      input = stream_in.read ().data;

      for (int j = 0; j < (CHANNEL_WIDTH / STATE_VECTOR_WIDTH); j++)
      {
#pragma HLS unroll
#pragma HLS pipeline
        if (i + j < vectorSize)
        {
#pragma HLS pipeline
          register_B.u32 = DATA16_TO_FLOAT32(input >> (STATE_VECTOR_WIDTH * j));
          state_vector[i + j] = register_B.f32;
        }
      }
    }

    sum = 0.0f;
    for (unsigned short spikeID = 0; spikeID < vectorSize; spikeID++)
    {
#pragma HLS pipeline
      if (sum < random_value)
      {
#pragma HLS pipeline
        sum += state_vector[spikeID];

        if (random_value <= sum || (spikeID == vectorSize - 1))
        {
#pragma HLS pipeline
          spike_matrix[ip_index] = spikeID;
        }
      }
    }

    for (int batch = 0; batch < kernelSize; batch++)
    {
#pragma HLS pipeline
      for (int i = 0; i < vectorSize; i += (CHANNEL_WIDTH / WEIGHT_VECTOR_WIDTH))
      {
#pragma HLS pipeline
        input = stream_in.read ().data;

        for (int j = 0; j < (CHANNEL_WIDTH / WEIGHT_VECTOR_WIDTH); j++)
        {
#pragma HLS unroll
#pragma HLS pipeline
          if (i + j < vectorSize)
          {
#pragma HLS pipeline
            register_B.u32 = DATA08_TO_FLOAT32(input >> (WEIGHT_VECTOR_WIDTH * j));
            weight_vector[i + j] = register_B.f32;
          }
        }
      }

      sum = 0.0f;
      for (int i = 0; i < vectorSize; i++)
      {
#pragma HLS pipeline
        temp_data[i] = state_vector[i] * weight_vector[i];
        sum += temp_data[i];
      }

      if (NEGLECTING_CONSTANT < sum)
      {
#pragma HLS pipeline
        epsion_over_sum = epsilon / sum;
        for (int i = 0; i < vectorSize; i++)
        {
#pragma HLS pipeline
          state_vector[i] = reverse_epsilon * (state_vector[i] + temp_data[i] * epsion_over_sum);
        }
      }
    }


    for (int i = 0; i < vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
    {
#pragma HLS pipeline
      for (int j = 0; j < (CHANNEL_WIDTH / STATE_VECTOR_WIDTH); j++)
      {
#pragma HLS unroll
#pragma HLS pipeline
        if (i + j < vectorSize)
        {
#pragma HLS pipeline
          register_A.f32 = state_vector[i + j];
          channel.data = (~(((ap_uint<CHANNEL_WIDTH> ) 0xFFFF) << (STATE_VECTOR_WIDTH * j)) & channel.data)
                         | (((ap_uint<CHANNEL_WIDTH> ) (FLOAT32_TO_DATA16(register_A.u32))) << (STATE_VECTOR_WIDTH * j));
        }
      }
      stream_out.write (channel);
    }
  }

  for (int i = 0; i < layerSize; i += (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH))
  {
#pragma HLS pipeline
    for (int j = 0; j < (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH); j++)
    {
#pragma HLS unroll
#pragma HLS pipeline
      if (i + j < layerSize)
      {
#pragma HLS pipeline
        channel.data = (~(((ap_uint<CHANNEL_WIDTH> )              0xFFFF) << (SPIKE_VECTOR_WIDTH * j)) & channel.data)
                       | (((ap_uint<CHANNEL_WIDTH> ) spike_matrix[i + j]) << (SPIKE_VECTOR_WIDTH * j));
      }
    }

    channel.last = ((i + (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH)) >= layerSize);
    stream_out.write (channel);
  }

  channel.last = 0;

  return debug_index;
}
