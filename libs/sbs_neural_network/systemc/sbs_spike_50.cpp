#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

typedef union
{
  unsigned int u32;
  float f32;
} FloatToInt;

#define VECTOR_SIZE     (50)

typedef ap_axis<32, 2, 5, 6> StreamChannel;

void sbs_spike_50 (hls::stream<StreamChannel> &stream_in,
                 hls::stream<StreamChannel> &stream_out,
                 int layerSize)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static int ip_index;
  static int spikeID;
  static StreamChannel channel;
  static float random_value;

  static FloatToInt float_to_int;

  static float sum;

  for (ip_index = 0; ip_index < layerSize; ip_index++)
  {
#pragma HLS pipeline
    if (ip_index == 0)
    {
      channel = stream_in.read ();
      float_to_int.u32 = channel.data;
    }
    else
    {
      float_to_int.u32 = stream_in.read ().data;
    }
    random_value = float_to_int.f32;

    sum = 0.0f;
    for (spikeID = 0; spikeID < VECTOR_SIZE; spikeID++)
    {
#pragma HLS pipeline
      float_to_int.u32 = stream_in.read ().data;
      if (sum < random_value)
      {
        sum += float_to_int.f32;
        if (random_value <= sum || (spikeID == VECTOR_SIZE - 1))
        {
          channel.last = (ip_index == layerSize - 1);
          channel.data = spikeID;
          stream_out.write(channel);
        }
      }
    }
  }
}
