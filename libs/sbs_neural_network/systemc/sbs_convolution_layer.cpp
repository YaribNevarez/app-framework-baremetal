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

#define MAX_VECTOR_SIZE             (64)
#define MAX_SPIKE_MATRIX_SIZE       (25*25)
#define MAX_INPUT_SPIKE_MATRIX_SIZE (5*5)

//static Data32 _d32;
#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0x0000FFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (0x000000FF & (((unsigned int)(d)) >> 19))

#define NEGLECTING_CONSTANT   ((float)1e-20)

#define CHANNEL_WIDTH         32
#define STATE_VECTOR_WIDTH    16
#define WEIGHT_VECTOR_WIDTH   8
#define SPIKE_VECTOR_WIDTH    16

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

unsigned int sbs_convolution_layer (hls::stream<StreamChannel> &stream_in,
                                    hls::stream<StreamChannel> &stream_out,
                                    int * debug,
                                    SbsHwMode mode)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out

#pragma HLS INTERFACE m_axi     port=debug       offset=slave    bundle=DEBUG
#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS

#pragma HLS INTERFACE s_axilite port=mode        bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  StreamChannel channel;
  ap_uint<CHANNEL_WIDTH> input;

  static SbsHardwareProfile hwProfile;

  static unsigned short input_spike_matrix[MAX_INPUT_SPIKE_MATRIX_SIZE];

  static ap_uint<WEIGHT_VECTOR_WIDTH> weight_matrix[52000];

  static unsigned short spike_matrix[MAX_SPIKE_MATRIX_SIZE];

  static float state_vector[MAX_VECTOR_SIZE];

  static float weight_vector[MAX_VECTOR_SIZE];

  static float temp_data[MAX_VECTOR_SIZE];


  float epsion_over_sum;
  float random_value;

  static Data32 register_A;
  static Data32 register_B;

  static float reverse_epsilon;
  float sum;

  unsigned int debug_index = 0;

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

        register_A.u32 = stream_in.read ().data;
        hwProfile.epsilon = register_A.f32;

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
            weight_matrix[i + j] = input >> (WEIGHT_VECTOR_WIDTH * j);
          }
        }
      }
      break;
    case SBS_HW_INFERENCE:
      {
        for (int ip_index = 0; ip_index < hwProfile.layerSize; ip_index++)
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

          for (int i = 0; i < hwProfile.vectorSize; i += (CHANNEL_WIDTH / STATE_VECTOR_WIDTH))
          {
  #pragma HLS pipeline
            input = stream_in.read ().data;

            for (int j = 0; j < (CHANNEL_WIDTH / STATE_VECTOR_WIDTH); j++)
            {
  #pragma HLS unroll
  #pragma HLS pipeline
              if (i + j < hwProfile.vectorSize)
              {
  #pragma HLS pipeline
                if (0xFFFF & (input >> (STATE_VECTOR_WIDTH * j)))
                {
  #pragma HLS pipeline
                  register_B.u32 = DATA16_TO_FLOAT32(input >> (STATE_VECTOR_WIDTH * j));
                  state_vector[i + j] = register_B.f32;
                }
                else
                {
  #pragma HLS pipeline
                  state_vector[i + j] = 0;
                }
              }
            }
          }

          sum = 0.0f;
          for (unsigned short spikeID = 0; spikeID < hwProfile.vectorSize; spikeID++)
          {
  #pragma HLS pipeline
            if (sum < random_value)
            {
  #pragma HLS pipeline
              sum += state_vector[spikeID];

              if (random_value <= sum || (spikeID == hwProfile.vectorSize - 1))
              {
  #pragma HLS pipeline
                spike_matrix[ip_index] = spikeID;
              }
            }
          }

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

          for (int batch = 0; batch < hwProfile.kernelSize; batch++)
          {
  #pragma HLS pipeline
            int tensor_index = input_spike_matrix[batch] * hwProfile.vectorSize
                + batch * hwProfile.weightDepth * hwProfile.vectorSize;

            for (int i = 0; i < hwProfile.vectorSize; i++)
            {
  #pragma HLS pipeline
              char weight = weight_matrix[tensor_index + i];
              if (0xFF & weight)
              {
  #pragma HLS pipeline
                register_B.u32 = DATA8_TO_FLOAT32(weight);
                weight_vector[i] = register_B.f32;
              }
              else
              {
  #pragma HLS pipeline
                weight_vector[i] = 0;
              }
            }

            sum = 0.0f;
            for (int i = 0; i < hwProfile.vectorSize; i++)
            {
  #pragma HLS pipeline
              if ((state_vector[i] != 0) && (weight_vector[i] != 0))
              {
  #pragma HLS pipeline
                temp_data[i] = state_vector[i] * weight_vector[i];
                sum += temp_data[i];
              }
              else
              {
  #pragma HLS pipeline
                temp_data[i] = 0;
              }
            }

            if (NEGLECTING_CONSTANT < sum)
            {
  #pragma HLS pipeline
              epsion_over_sum = hwProfile.epsilon / sum;
              for (int i = 0; i < hwProfile.vectorSize; i++)
              {
  #pragma HLS pipeline
                if (temp_data[i] != 0)
                {
  #pragma HLS pipeline
                  state_vector[i] = reverse_epsilon
                      * (state_vector[i] + temp_data[i] * epsion_over_sum);
                }
                else if (state_vector[i] != 0)
                {
  #pragma HLS pipeline
                  state_vector[i] = reverse_epsilon * state_vector[i];
                }
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
                register_A.f32 = state_vector[i + j];
                if ((register_A.u32 & 0xf0000000) == 0x30000000)
                {
  #pragma HLS pipeline
                  channel.data = (~(((ap_uint<CHANNEL_WIDTH> ) 0xFFFF)
                      << (STATE_VECTOR_WIDTH * j)) & channel.data)
                      | (((ap_uint<CHANNEL_WIDTH> ) (FLOAT32_TO_DATA16(
                          register_A.u32))) << (STATE_VECTOR_WIDTH * j));
                }
                else
                {
  #pragma HLS pipeline
                  channel.data = (~(((ap_uint<CHANNEL_WIDTH> ) 0xFFFF) << (STATE_VECTOR_WIDTH * j)) & channel.data);
                }
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
