#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

typedef ap_axis<32, 2, 5, 6> StreamChannel;

#define MAX_VECTOR_SIZE   50

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0xFFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (  0xFF & (((unsigned int)(d)) >> 19))

#define DATA_SIZE     sizeof(short) // Bytes
#define BUFFER_SIZE   ((MAX_VECTOR_SIZE + 1) * DATA_SIZE) / sizeof(Data32) + 1

void sbs_spike_50 (hls::stream<StreamChannel> &stream_in,
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

  static float data[MAX_VECTOR_SIZE];
  static unsigned int index;
  unsigned char index_spike;
  unsigned int debug_flags;

  static int ip_index;
  static unsigned short spikeID;
  static StreamChannel channel;
  static float random_value;

  static Data32 register_A;
  static Data32 register_B;
  unsigned int temp;

  static float sum;

  index_spike = 0;
  temp = 0;

  debug_flags = 0;

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
      register_A.u32 = stream_in.read ().data;

      register_B.u32 = DATA16_TO_FLOAT32(register_A.u32 >> 0);
      data[index] = register_B.f32;
      index ++;

      if (index < vectorSize)
      {
        register_B.u32 = DATA16_TO_FLOAT32(register_A.u32 >> 16);
        data[index] = register_B.f32;
        index++;
      }
    }

    sum = 0.0f;
    for (spikeID = 0; spikeID < vectorSize; spikeID++)
    {
#pragma HLS pipeline
      if (sum < random_value)
      {
        sum += data[spikeID];
        if ((random_value <= sum) || (spikeID == vectorSize - 1))
        {
          debug_flags ++;
          *debug = debug_flags;

          temp |= ((unsigned int)spikeID) << (16 * index_spike);
          index_spike ++;

          if ((index_spike == 2) || (ip_index == layerSize - 1))
          {
            channel.last = (ip_index == layerSize - 1);
            channel.data = temp;
            stream_out.write (channel);

            index_spike = 0;
            temp = 0;
          }
        }
      }
    }
  }
}
