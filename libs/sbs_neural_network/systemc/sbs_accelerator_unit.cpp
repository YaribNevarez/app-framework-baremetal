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

#define BUILD_FLOAT(s, exponent, mantissa) ((0x80000000 & ((s) << 31)) | (0x7f800000 & (((exponent) + 0x7f) << 23)) | ((mantissa) & 0x7FFFFF))

#define DATA08_GET_EXPONENT(x) ((0x70 | (0x0F & ((x) >> 4))) - 0x7F)

#define DATA16_GET_EXPONENT(x) ((0x60 | ((x) >> 11 )) - 0x7F)
#define DATA16_GET_MANTISSA(x) ((0x800 | (0x7FF & (x))) << 12)

#define DATA32_GET_EXPONENT(x) ((0xFF & ((x) >> 23)) - 0x7F)
#define DATA32_GET_MANTISSA(x) (0x00800000 | ((0x7FFFFF) & (x)))

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

#define UPDATE_CUSTOM_FLOAT true
#define SPIKE_CUSTOM_FLOAT  false

unsigned int sbs_accelerator_unit (hls::stream<StreamChannel> &stream_in,
                                   hls::stream<StreamChannel> &stream_out,
                                   int * debug,
                                   int flags,
                                   int layerSize,
                                   int kernelSize,
                                   int vectorSize,
                                   float epsilon)
{
//#pragma HLS INTERFACE m_axi     port=debug        offset=slave    bundle=DEBUG

#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=flags       bundle=CRTL_BUS
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

  /////////////////////////////////////////////////////////////////////////////
  ap_uint<64> state_vector_magnitude[MAX_VECTOR_SIZE];
  ap_uint<64> random_value;
  ap_int<8>   exponent;
  ap_uint<64> mantissa;
  ap_uint<64> sum_magnitude;
  /////////////////////////////////////////////////////////////////////////////

  static ap_uint<WEIGHT_VECTOR_WIDTH> weight_vector[MAX_VECTOR_SIZE];
//#pragma HLS array_partition variable=weight_vector block factor=246

  static float temp_data[MAX_VECTOR_SIZE];
//#pragma HLS array_partition variable=temp_data block factor=246

  static float epsion_over_sum;

  static ap_uint<CHANNEL_WIDTH> input;

  static Data32 register_A;
  static Data32 register_B;
  static Data32 data;
  Data32 hw;
  Data32 weight_data;

  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  Data32 sum;
  Data32 sum_spike;
  Data32 random_value_float;
  unsigned short spikeID;

  unsigned int debug_index = 0;
  /////////////////////////////////////////////////////////////////////////////
  ap_int<32>  w_exponent;

  ap_int<32>  h_exponent;
  ap_uint<64> h_mantissa;

  ap_int<32>  hw_exponent;
  ap_uint<64> hw_mantissa;

  ap_uint<8> w;
  /////////////////////////////////////////////////////////////////////////////

  channel.keep = -1;
  channel.strb = -1;

#if MT19937_HW
  if (!MT19937_initialized (0))
  {
    MT19937_sgenrand (0, 666);
  }
#endif

  LAYER_UPDATE: for (int ip_index = 0; ip_index < layerSize; ip_index++)
  {
#if MT19937_HW
    random_value = ((float) MT19937_rand (0)) / ((float) 0xFFFFFFFF);
#else
#pragma HLS pipeline

#if SPIKE_CUSTOM_FLOAT
    random_value = ((ap_uint<64> ) stream_in.read ().data) << (32 - 9);
#else
    random_value_float.u32 = stream_in.read ().data;
#endif

#endif

    VECTOR_INPUT: for (int i = 0; i < vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
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
#if SPIKE_CUSTOM_FLOAT
          exponent = DATA16_GET_EXPONENT(input >> (STATE_VECTOR_WIDTH * j));
          mantissa = ((ap_uint<64> ) DATA16_GET_MANTISSA(input >> (STATE_VECTOR_WIDTH * j))) << 32;
          if (exponent < 0)
            state_vector_magnitude[i + j] = mantissa >> -exponent;
          else
            state_vector_magnitude[i + j] = mantissa << exponent;
#endif
        }
      }
    }

#if SPIKE_CUSTOM_FLOAT
    sum_magnitude = 0;
    SPIKE_FIRE: for (spikeID = 0;
        (sum_magnitude < random_value) && (spikeID < vectorSize); spikeID++)
    {
#pragma HLS pipeline
      sum_magnitude += state_vector_magnitude[spikeID];
    }
#else
    sum_spike.u32 = 0;
    SPIKE_FIRE: for (spikeID = 0;
        (sum_spike.f32 < random_value_float.f32) && (spikeID < vectorSize);
        spikeID++)
    {
#pragma HLS pipeline
      sum_spike.f32 += state_vector[spikeID];
    }
#endif

    spike_matrix[ip_index] = (0 < spikeID) ? spikeID - 1 : 0;

    BATCH_UPDATE: for (int batch = 0; batch < kernelSize; batch++)
    {
#pragma HLS pipeline
      WEIGHT_INPUT: for (int i = 0; i < vectorSize; i += (CHANNEL_WIDTH / WEIGHT_VECTOR_WIDTH))
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
            weight_vector[i + j] = 0xFF & (input >> (WEIGHT_VECTOR_WIDTH * j));
          }
        }
      }

      sum_magnitude = 0;
      sum.u32 = 0;
      DOT_PRODUCT: for (int i = 0; i < vectorSize; i++)
      {
#pragma HLS pipeline
        data.f32 = state_vector[i];

        if (data.u32)
        {
          w = weight_vector[i];
#if UPDATE_CUSTOM_FLOAT
          h_exponent = DATA32_GET_EXPONENT(data.u32);
          h_mantissa = DATA32_GET_MANTISSA(data.u32);

          w_exponent = DATA08_GET_EXPONENT(w);

          hw_exponent = h_exponent + w_exponent;

          hw_mantissa = (h_mantissa << (32 - 4)) * (0x10 | (0x0F & w));

          if (hw_mantissa & 0x0100000000000000)
          {
            hw_exponent++;
            hw_mantissa >>= 1;
          }

          hw.u32 = BUILD_FLOAT(0, hw_exponent, hw_mantissa >> 32);

          if (hw_exponent < 0)
            sum_magnitude += hw_mantissa >> -hw_exponent;
          else
            sum_magnitude += hw_mantissa << hw_exponent;

#else
          weight_data.u32 = DATA08_TO_FLOAT32(w);

          hw.f32 = data.f32 * weight_data.f32;
          sum.f32 += hw.f32;
#endif


        }
        else
        {
          hw.u32 = 0;
        }

        temp_data[i] = hw.f32;
      }

#if UPDATE_CUSTOM_FLOAT
      if (sum_magnitude)
#else
      if (sum.u32)
#endif
      {
#pragma HLS pipeline

#if UPDATE_CUSTOM_FLOAT
        NORMALIZE_SUM: for (exponent = 0; !(0x80000000000000 & sum_magnitude); exponent++)
        { // Normalize
#pragma HLS pipeline
          sum_magnitude <<= 1;
        }

        sum.u32 = BUILD_FLOAT(0, -exponent, (sum_magnitude >> 32));
#endif

        epsion_over_sum = epsilon / sum.f32;
        VECTOR_UPDATE: for (int i = 0; i < vectorSize; i++)
        {
#pragma HLS pipeline
          state_vector[i] = reverse_epsilon * (state_vector[i] + temp_data[i] * epsion_over_sum);
        }
      }
    }


    VECTOR_OUTPUT: for (int i = 0; i < vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
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

  SPIKE_OUTPUT: for (int i = 0; i < layerSize; i += (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH))
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
