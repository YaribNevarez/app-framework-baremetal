#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

#define MT19937_HW true

#if MT19937_HW

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

#define MT19937_SEED 666

static unsigned int MT19937_rand (void)
{
#pragma HLS inline off
  static unsigned int mt[N]; /* the array for the state vector  */
#pragma HLS ARRAY_PARTITION variable=mt block factor=1 dim=1

  static unsigned int mti = N + 1; /* mti==N+1 means mt[N] is not initialized */

  unsigned int y;
  static unsigned int mag01[2] = { 0x0, MATRIX_A };
#pragma HLS array_partition variable=mag01 complete
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

  if (mti >= N)
  { /* generate N words at one time */
    int kk;

    if (mti == N + 1)
    {
	  mt[0] = MT19937_SEED & 0xffffffff;
	  MT19937_INITIALIZE_A:for (mti = 1; mti < N; mti++)
	  {
#pragma HLS pipeline
		mt[mti] = (69069 * mt[mti - 1]) & 0xffffffff;
	  }
    }

    MT19937_INITIALIZE_B: for (kk = 0; kk < N - M; kk++)
    {
#pragma HLS pipeline
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
    }

    MT19937_INITIALIZE_C: for (; kk < N - 1; kk++)
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

#define MANTISSA_BIT_WIDTH    23
#define MT19937_BIT_WIDTH     32

#define CHANNEL_WIDTH         32
#define STATE_VECTOR_WIDTH    16
#define SPIKE_VECTOR_WIDTH    16

#define SPIKE_COUNT_MASK      ((CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH) - 1)

typedef ap_axis<CHANNEL_WIDTH, 2, 5, 6> StreamChannel;

#define MAX_VECTOR_SIZE   50

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

void sbs_spike_unit (hls::stream<StreamChannel> &stream_in,
                     hls::stream<StreamChannel> &stream_out,
                     int * debug,
                     int layerSize,
                     int vectorSize)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  /////////////////////////////////////////////////////////////////////////////
  ap_uint<64> state_vector_magnitude[MAX_VECTOR_SIZE];
  ap_uint<64> random_value;
  ap_int<8>   exponent;
  ap_uint<64> mantissa;
  ap_uint<64> sum_magnitude;
  /////////////////////////////////////////////////////////////////////////////
#pragma HLS array_partition variable=state_vector_magnitude complete

  unsigned int debug_flags;

  static StreamChannel channel;

  static ap_uint<CHANNEL_WIDTH> input;
  static Data32 register_B;

  channel.keep = -1;
  channel.strb = -1;

  LAYER_SPIKE: for (int ip_index = 0; ip_index < layerSize; ip_index++)
  {
    random_value = ((ap_uint<64> ) MT19937_rand ()) << (32 - (MT19937_BIT_WIDTH - MANTISSA_BIT_WIDTH));

    VECTOR_INPUT: for (int index = 0; index < vectorSize; index += CHANNEL_WIDTH / STATE_VECTOR_WIDTH)
    {
#pragma HLS pipeline
      input = stream_in.read ().data;

      for (int i = 0; i < CHANNEL_WIDTH / STATE_VECTOR_WIDTH; i++)
      {
#pragma HLS unroll
#pragma HLS pipeline
        if (index + i < vectorSize)
        {
#pragma HLS pipeline
          /////////////////////////////////////////////////////////////////////////////
          exponent = DATA16_GET_EXPONENT(input >> (STATE_VECTOR_WIDTH * i));
          mantissa = ((ap_uint<64> ) DATA16_GET_MANTISSA(input >> (STATE_VECTOR_WIDTH * i))) << 32;
          if (exponent < 0)
            state_vector_magnitude[index + i] = mantissa >> -exponent;
          else
            state_vector_magnitude[index + i] = mantissa << exponent;
          /////////////////////////////////////////////////////////////////////////////
        }
      }
    }

    sum_magnitude = 0;
    SPIKE_OUTPUT: for (short spikeID = 0; (sum_magnitude < random_value) && (spikeID < vectorSize); spikeID++)
    {
#pragma HLS pipeline
      sum_magnitude += state_vector_magnitude[spikeID];
      if ((random_value <= sum_magnitude) || (spikeID == vectorSize - 1))
      {
#pragma HLS pipeline
        channel.data =
         (~(((ap_uint<CHANNEL_WIDTH> )  0xFFFF) << (SPIKE_VECTOR_WIDTH * (ip_index & SPIKE_COUNT_MASK))) & channel.data)
        | (((ap_uint<CHANNEL_WIDTH> )  spikeID) << (SPIKE_VECTOR_WIDTH * (ip_index & SPIKE_COUNT_MASK)));

        if (((ip_index & SPIKE_COUNT_MASK) == SPIKE_COUNT_MASK) || (ip_index == layerSize - 1))
        {
#pragma HLS pipeline
          channel.last = (ip_index == layerSize - 1);

          stream_out.write (channel);
        }
      }
    }
  }
}
