#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

#define MAX_VECTOR_SIZE       (1024)
#define MAX_SPIKE_MATRIX_SIZE (60*60)

//static Data32 _d32;
#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0x0000FFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (0x000000FF & (((unsigned int)(d)) >> 19))

#define NEGLECTING_CONSTANT   ((float)1e-20)

typedef ap_axis<32, 2, 5, 6> StreamChannel;

static int debug_flags;


void sbs_accelerator (hls::stream<StreamChannel> &stream_in,
                      hls::stream<StreamChannel> &stream_out,
                      int * debug,
                      int layerSize,
                      int kernelSize,
                      int vectorSize,
                      float epsilon)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=debug       bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=kernelSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=epsilon     bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static unsigned int index;
  static unsigned int index_channel;

  static int ip_index;
  static unsigned short spikeID;
  static int i;
  static int batch;
  static StreamChannel channel;

  static unsigned short spike_matrix[MAX_SPIKE_MATRIX_SIZE];
#pragma HLS array_partition variable=spike_matrix block factor=4

  static int spike_index;

  static float state_vector[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=state_vector block factor=246

  static float weight_vector[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=weight_vector block factor=246

  static float temp_data[MAX_VECTOR_SIZE];
#pragma HLS array_partition variable=temp_data block factor=246

  static float epsion_over_sum;
  static float random_value;

  static unsigned int temp;

  static Data32 register_A;
  static Data32 register_B;

  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  static float sum;

  temp = 0;
  index_channel = 0;
  debug_flags = 1;

  for (ip_index = 0; ip_index < layerSize; ip_index++)
  {
#pragma HLS pipeline
    if (ip_index == 0)
    {
      channel = stream_in.read ();
      register_A.u32 = channel.data;
    }
    else
    {
      register_A.u32 = stream_in.read ().data;
    }
    random_value = register_A.f32;

    index = 0;
    while (index < vectorSize)
    {
#pragma HLS pipeline
      i = stream_in.read ().data;

      register_B.u32 = DATA16_TO_FLOAT32(i >> 0);
      state_vector[index] = register_B.f32;
      index ++;

      if (index < vectorSize)
      {
        register_B.u32 = DATA16_TO_FLOAT32(i >> 16);
        state_vector[index] = register_B.f32;
        index++;
      }
    }

    sum = 0.0f;
    for (spikeID = 0; spikeID < vectorSize; spikeID++)
    {
#pragma HLS pipeline
      if (sum < random_value)
      {
        sum += state_vector[spikeID];

        if (random_value <= sum || (spikeID == vectorSize - 1))
        {
          spike_matrix[ip_index] = spikeID;
        }
      }
    }

    for (batch = 0; batch < kernelSize; batch++)
    {
#pragma HLS pipeline
      index = 0;
      while (index < vectorSize)
      {
  #pragma HLS pipeline
        i = stream_in.read ().data;

        register_B.u32 = DATA8_TO_FLOAT32(i >> 0);
        weight_vector[index] = register_B.f32;
        index++;

        if (index < vectorSize)
        {
          register_B.u32 = DATA8_TO_FLOAT32(i >> 8);
          weight_vector[index] = register_B.f32;
          index++;
        }

        if (index < vectorSize)
        {
          register_B.u32 = DATA8_TO_FLOAT32(i >> 16);
          weight_vector[index] = register_B.f32;
          index++;
        }

        if (index < vectorSize)
        {
          register_B.u32 = DATA8_TO_FLOAT32(i >> 24);
          weight_vector[index] = register_B.f32;
          index++;
        }
      }

      sum = 0.0f;
      for (i = 0; i < vectorSize; i++)
      {
#pragma HLS pipeline
        temp_data[i] = state_vector[i] * weight_vector[i];
        sum += temp_data[i];
      }

      if (NEGLECTING_CONSTANT < sum)
      {
        epsion_over_sum = epsilon / sum;
        for (i = 0; i < vectorSize; i++)
        {
#pragma HLS pipeline
          state_vector[i] = reverse_epsilon
              * (state_vector[i] + temp_data[i] * epsion_over_sum);
        }
      }
    }


    for (i = 0; i < vectorSize; i++)
    {
#pragma HLS pipeline
      register_A.f32 = state_vector[i];

      if ((register_A.u32 & 0xf0000000) == 0x30000000)
      {
        temp |= (FLOAT32_TO_DATA16(register_A.u32)) << (16 * index_channel);
      }

      index_channel ++;

      if (index_channel == 2)
      {
        channel.data = temp;
        stream_out.write (channel);
        index_channel = 0;
        temp = 0;
      }
    }
  }

  index_channel = 0;
  temp = 0;
  for (i = 0; i < layerSize; i++)
  {
#pragma HLS pipeline
    temp |= ((unsigned int)spike_matrix[i]) << (16 * index_channel);
    index_channel ++;

    if ((index_channel == 2) || (i == layerSize - 1))
    {
      channel.data = temp;
      channel.last = (i == layerSize - 1);
      stream_out.write (channel);
      index_channel = 0;
      temp = 0;
    }
  }
}
