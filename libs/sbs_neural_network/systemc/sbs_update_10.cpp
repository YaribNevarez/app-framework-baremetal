#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

typedef union
{
  unsigned int u32;
  float f32;
} FloatToInt;

#define VECTOR_SIZE           (10)

#define NEGLECTING_CONSTANT   ((float)1e-20)

typedef ap_axis<32, 2, 5, 6> StreamChannel;


void sbs_update_10 (hls::stream<StreamChannel> &stream_in,
                 hls::stream<StreamChannel> &stream_out,
                 float epsilon)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=epsilon     bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static int i;
  static int batch;
  static StreamChannel channel;
  static int spike;
  static int spike_index;
  static float state_vector[VECTOR_SIZE];
  static float temp_data[VECTOR_SIZE];
  static float epsion_over_sum;
  static float random_value;

  static FloatToInt float_to_int;

  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  static float sum;

#pragma HLS pipeline
  channel = stream_in.read ();
  float_to_int.u32 = channel.data;
  random_value = float_to_int.f32;

  sum = 0.0f;
  for (i = 0; i < VECTOR_SIZE; i++)
  {
#pragma HLS pipeline
    float_to_int.u32 = stream_in.read ().data;
    state_vector[i] = float_to_int.f32;
    if (sum < random_value)
    {
      sum += state_vector[i];
      if (random_value <= sum || (i == VECTOR_SIZE - 1))
      {
        spike = i;
      }
    }
  }


  sum = 0.0f;
  for (i = 0; i < VECTOR_SIZE; i++)
  {
#pragma HLS pipeline
    float_to_int.u32 = stream_in.read ().data;
    temp_data[i] = state_vector[i] * float_to_int.f32;
    sum += temp_data[i];
  }

  if (NEGLECTING_CONSTANT < sum)
  {
    epsion_over_sum = epsilon / sum;
    for (i = 0; i < VECTOR_SIZE; i++)
    {
#pragma HLS pipeline
      state_vector[i] = reverse_epsilon
          * (state_vector[i] + temp_data[i] * epsion_over_sum);
    }
  }

  for (i = 0; i < VECTOR_SIZE; i++)
  {
#pragma HLS pipeline
    float_to_int.f32 = state_vector[i];
    channel.data = float_to_int.u32;
    stream_out.write (channel);
  }

  channel.data = spike;
  channel.last = 1;
  stream_out.write (channel);
}
