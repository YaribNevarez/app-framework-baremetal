#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"


#define UPDATE_CUSTOM_FLOAT       true
#define SPIKE_CUSTOM_FLOAT        false

#define WEIGHT_EXPONENT_BIT_WIDTH 4
#define WEIGHT_MANTISSA_BIT_WIDTH 1

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


typedef enum
{
  SBS_HW_INITIALIZE,
  SBS_HW_INFERENCE
} SbsHwMode;


typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

#define MAX_VECTOR_SIZE             (32)
#define MAX_SPIKE_MATRIX_SIZE       (25*25)
#define MAX_INPUT_SPIKE_MATRIX_SIZE (5*5)

#define BUILD_FLOAT(s, exponent, mantissa) ((0x80000000 & ((s) << 31)) | (0x7f800000 & (((exponent) + 0x7f) << 23)) | ((mantissa) & 0x7FFFFF))

#define DATA16_GET_EXPONENT(x) ((0x60 | ((x) >> 11 )) - 0x7F)
#define DATA16_GET_MANTISSA(x) ((0x800 | (0x7FF & (x))) << 12)

#define DATA08_GET_EXPONENT(x) ((0x70 | (0x0F & ((x) >> WEIGHT_MANTISSA_BIT_WIDTH))) - 0x7F)

#define DATA04_GET_EXPONENT(x) ((0x70 | (0x0F & (x))) - 0x7F)

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

#define WEIGHT_BIT_WIDTH      (WEIGHT_EXPONENT_BIT_WIDTH + WEIGHT_MANTISSA_BIT_WIDTH)
#define WEIGHT_MATRIX_SIZE    1600

typedef ap_axis<CHANNEL_WIDTH, 2, 5, 6> StreamChannel;

typedef struct
{
  int layerSize;
  int kernelSize;
  int vectorSize;
  float epsilon;

  int weightRows;
  int weightColumns;
  int weightDepth;
} SbsHardwareProfile;

unsigned int sbs_conv_layer_32 (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    int flags,
                                    int * debug,
                                    SbsHwMode mode)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out

#pragma HLS INTERFACE s_axilite port=flags       bundle=CRTL_BUS

//#pragma HLS INTERFACE m_axi     port=debug       offset=slave    bundle=DEBUG
#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS

#pragma HLS INTERFACE s_axilite port=mode        bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  StreamChannel channel;
  ap_uint<CHANNEL_WIDTH> input;

  static SbsHardwareProfile hwProfile;
  static unsigned int       weightDepthSize;

  static unsigned short input_spike_matrix[MAX_INPUT_SPIKE_MATRIX_SIZE];
//#pragma HLS ARRAY_PARTITION variable=input_spike_matrix complete dim=1

  static ap_uint<WEIGHT_BIT_WIDTH> weight_matrix[WEIGHT_MATRIX_SIZE];
//#pragma HLS ARRAY_PARTITION variable=weight_matrix block factor=1 dim=1

  static unsigned short spike_matrix[MAX_SPIKE_MATRIX_SIZE];
//#pragma HLS ARRAY_PARTITION variable=spike_matrix block factor=32 dim=1

  static float state_vector[MAX_VECTOR_SIZE];
//#pragma HLS ARRAY_PARTITION variable=state_vector complete dim=1

  /////////////////////////////////////////////////////////////////////////////
  ap_uint<64> state_vector_magnitude[MAX_VECTOR_SIZE];
  ap_uint<64> random_value;
  ap_int<8>   exponent;
  ap_uint<64> mantissa;
  ap_uint<64> sum_magnitude;
  /////////////////////////////////////////////////////////////////////////////

//  static float weight_vector[MAX_VECTOR_SIZE];
//#pragma HLS ARRAY_PARTITION variable=weight_vector block factor=32 dim=1

  static float temp_data[MAX_VECTOR_SIZE];
//#pragma HLS ARRAY_PARTITION variable=temp_data block factor=32 dim=1


  float epsion_over_sum;

  Data32 stream_data;
  Data32 data;
  Data32 weight_data;
  Data32 hw;

  static float reverse_epsilon;
  static float low_pass_epsilon;
  Data32 sum;
  Data32 sum_spike;
  Data32 random_value_float;
  unsigned short spikeID;

  unsigned int debug_index = 0;

  /////////////////////////////////////////////////////////////////////////////
  ap_uint<WEIGHT_BIT_WIDTH> weight;
  
  ap_uint<4> weight_and_mask;
  ap_uint<4> weight_or_mask;

  ap_int<8>  w_exponent;
  ap_uint<5>  w_mantissa;

  ap_int<8>  h_exponent;
  ap_uint<64> h_mantissa;

  ap_int<8>  hw_exponent;
  ap_uint<64> hw_mantissa;
/////////////////////////////////////////////////////////////////////////////

  weight_and_mask = (0xF & flags);
  weight_or_mask = 0xF & (flags >> 4);

  channel.keep = -1;
  channel.strb = -1;

#if MT19937_HW
  if (!MT19937_initialized (0))
  {
    MT19937_sgenrand (0, 666);
  }
#endif

  switch (mode)
  {
    case SBS_HW_INITIALIZE:
      {
        int data_size;

        hwProfile.layerSize = stream_in.read ().data;
        hwProfile.kernelSize = stream_in.read ().data;
        hwProfile.vectorSize = stream_in.read ().data;

        stream_data.u32 = stream_in.read ().data;
        hwProfile.epsilon = stream_data.f32;

        hwProfile.weightRows = stream_in.read ().data;
        hwProfile.weightColumns = stream_in.read ().data;
        hwProfile.weightDepth = stream_in.read ().data;

        data_size = hwProfile.weightRows
                  * hwProfile.weightColumns
                  * hwProfile.weightDepth
                  * hwProfile.vectorSize;

        reverse_epsilon = 1.0f / (1.0f + hwProfile.epsilon);

        for (int i = 0; i < data_size; i += (CHANNEL_WIDTH / WEIGHT_VECTOR_WIDTH))
        {
#pragma HLS pipeline
          input = stream_in.read ().data;
          for (int j = 0; j < (CHANNEL_WIDTH / WEIGHT_VECTOR_WIDTH); j++)
          {
#pragma HLS unroll
#pragma HLS pipeline
            weight_matrix[i + j] = input >> ((WEIGHT_VECTOR_WIDTH * j) + (WEIGHT_VECTOR_WIDTH - WEIGHT_BIT_WIDTH));
          }
        }

        weightDepthSize = hwProfile.weightDepth * hwProfile.vectorSize;
      }
      break;
    case SBS_HW_INFERENCE:
      {
        LAYER_UPDATE: for (int ip_index = 0; ip_index < hwProfile.layerSize; ip_index++)
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

          STATE_VECTOR_LOADING: for (int i = 0; i < hwProfile.vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
          {
  #pragma HLS pipeline
            input = stream_in.read ().data;

            STATE_VECTOR_LOADING_UNROLL: for (int j = 0; j < (CHANNEL_WIDTH / STATE_VECTOR_WIDTH); j++)
            {
  #pragma HLS unroll
  #pragma HLS pipeline
              if (i + j < hwProfile.vectorSize)
              {
  #pragma HLS pipeline
                data.u32 = DATA16_TO_FLOAT32(input >> (STATE_VECTOR_WIDTH * j));
                state_vector[i + j] = data.f32;
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
          SPIKE_GENERATION: for (spikeID = 0;
              (sum_magnitude < random_value) && (spikeID < hwProfile.vectorSize);
              spikeID++)
          {
  #pragma HLS pipeline
            sum_magnitude += state_vector_magnitude[spikeID];
          }
#else
          sum_spike.u32 = 0;
          SPIKE_GENERATION: for (spikeID = 0;
              (sum_spike.f32 < random_value_float.f32) && (spikeID < hwProfile.vectorSize);
              spikeID++)
          {
  #pragma HLS pipeline
            sum_spike.f32 += state_vector[spikeID];
          }
#endif

          spike_matrix[ip_index] = (0 < spikeID) ? spikeID - 1 : 0;

          for (int i = 0; i < hwProfile.kernelSize; i += (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH))
          {
  #pragma HLS pipeline
            input = stream_in.read ().data;
            for (int j = 0; j < (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH); j++)
            {
  #pragma HLS unroll
              if (i + j < hwProfile.kernelSize)
              {
  #pragma HLS pipeline
                input_spike_matrix[i + j] = input >> (SPIKE_VECTOR_WIDTH * j);
              }
            }
          }

          for (int batch = 0, weight_matrix_index = 0;
               batch < hwProfile.kernelSize;
               batch++, weight_matrix_index += weightDepthSize)
          {
  #pragma HLS pipeline
            int tensor_index = input_spike_matrix[batch] * hwProfile.vectorSize + weight_matrix_index;

            sum_magnitude = 0;
            sum.u32 = 0;
            HW_MUL: for (int i = 0; i < hwProfile.vectorSize; i++)
            {
#pragma HLS pipeline
              data.f32 = state_vector[i];

              if (data.u32)
              {
                weight = weight_matrix[tensor_index + i];
#if UPDATE_CUSTOM_FLOAT
                h_exponent = DATA32_GET_EXPONENT(data.u32);
                h_mantissa = DATA32_GET_MANTISSA(data.u32);

                w_exponent = DATA08_GET_EXPONENT(weight);
                w_mantissa = 0x10 | (0xF & ((weight & ((1 << WEIGHT_MANTISSA_BIT_WIDTH) - 1)) << (4 - WEIGHT_MANTISSA_BIT_WIDTH)));

                hw_exponent = h_exponent + w_exponent;

                hw_mantissa = (h_mantissa << (32 - 4)) * w_mantissa;

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
                weight_data.u32 = DATA08_TO_FLOAT32(weight);

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

              sum.u32 = BUILD_FLOAT(0, -exponent, sum_magnitude >> 32);
#endif

              epsion_over_sum = hwProfile.epsilon / sum.f32;
              low_pass_epsilon = reverse_epsilon * epsion_over_sum;
              update_loop: for (int i = 0; i < hwProfile.vectorSize; i++)
              {
  #pragma HLS pipeline
                state_vector[i] = reverse_epsilon * state_vector[i] + low_pass_epsilon * temp_data[i];
              }
            }
          }


          for (int i = 0; i < hwProfile.vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
          {
  #pragma HLS pipeline
            for (int j = 0; j < (CHANNEL_WIDTH / STATE_VECTOR_WIDTH); j++)
            {
  #pragma HLS unroll
  #pragma HLS pipeline
              if (i + j < hwProfile.vectorSize)
              {
  #pragma HLS pipeline
                data.f32 = state_vector[i + j];
                channel.data = (~(((ap_uint<CHANNEL_WIDTH> ) 0xFFFF) << (STATE_VECTOR_WIDTH * j)) & channel.data) | (((ap_uint<CHANNEL_WIDTH> ) (FLOAT32_TO_DATA16(data.u32))) << (STATE_VECTOR_WIDTH * j));
              }
            }
            stream_out.write (channel);
          }
        }

        for (int i = 0; i < hwProfile.layerSize; i += (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH))
        {
  #pragma HLS pipeline
          for (int j = 0; j < (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH); j++)
          {
  #pragma HLS unroll
  #pragma HLS pipeline
            if (i + j < hwProfile.layerSize)
            {
  #pragma HLS pipeline
              channel.data = (~(((ap_uint<CHANNEL_WIDTH> )              0xFFFF) << (SPIKE_VECTOR_WIDTH * j)) & channel.data)
                             | (((ap_uint<CHANNEL_WIDTH> ) spike_matrix[i + j]) << (SPIKE_VECTOR_WIDTH * j));
            }
          }

          channel.last = ((i + (CHANNEL_WIDTH / SPIKE_VECTOR_WIDTH)) >= hwProfile.layerSize);
          stream_out.write (channel);
        }

        channel.last = 0;
      }
      break;
    default:;
  }

  return debug_index;
}
