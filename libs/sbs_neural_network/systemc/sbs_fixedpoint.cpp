#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

#define H_QF    (21)
#define W_QF    (21)
#define H_MAX   (((unsigned int)1 << H_QF) - 1)
#define W_MAX   (((unsigned int)1 << W_QF) - 1)

#define MAX_VECTOR_SIZE       (1024)
#define MAX_SPIKE_MATRIX_SIZE (60*60)

typedef ap_axis<32, 2, 5, 6> StreamChannel;

//typedef ap_uint<H_QF>        Data1QF;
//typedef ap_uint<2*H_QF>      Data2QF;
//typedef ap_ufixed<H_QF,H_QF>        Data1QF;
//typedef ap_ufixed<2*H_QF,2*H_QF>      Data2QF;
//typedef ap_ufixed<3*H_QF,3*H_QF>      Data3QF;
typedef unsigned int    Data1QF;
typedef unsigned long   Data2QF;

void sbs_fixedpoint (hls::stream<StreamChannel> &stream_in,
                 hls::stream<StreamChannel> &stream_out,
                 unsigned int layerSize,
                 unsigned int kernelSize,
                 unsigned int vectorSize,
                 unsigned int epsilon)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=kernelSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=epsilon     bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static unsigned int ip_index;
  static unsigned int i;
  static unsigned int batch;
  static StreamChannel channel;
  static unsigned int spike_matrix[MAX_SPIKE_MATRIX_SIZE];
  static unsigned int spike_index;
  static Data1QF state_vector[MAX_VECTOR_SIZE];
  static Data2QF temp_data[MAX_VECTOR_SIZE];
  static Data2QF epsion_over_sum;
  static Data1QF random_value;

  Data2QF sum;

  Data2QF reverse_epsilon = (((Data2QF)H_MAX) << H_QF);

  reverse_epsilon = reverse_epsilon / (Data2QF)((Data2QF)H_MAX + (Data2QF)(epsilon));

  for (ip_index = 0; ip_index < layerSize; ip_index++)
  {
#pragma HLS pipeline
    if (ip_index == 0)
    {
      channel = stream_in.read ();
      random_value = channel.data;
    }
    else
    {
      random_value = stream_in.read ().data;
    }

    sum = 0;
    for (i = 0; i < vectorSize; i++)
    {
#pragma HLS pipeline
      state_vector[i] = stream_in.read ().data;
      if (sum < random_value)
      {
        sum += state_vector[i];
        if (random_value <= sum || (i == vectorSize - 1))
        {
          spike_matrix[ip_index] = i;
        }
      }
    }

    for (batch = 0; batch < kernelSize; batch++)
    {
#pragma HLS pipeline
      sum = 0;
      for (i = 0; i < vectorSize; i++)
      {
#pragma HLS pipeline
        temp_data[i] = state_vector[i];
        temp_data[i] *=  (stream_in.read ().data << (H_QF - W_QF));
        sum += temp_data[i];
      }

      if (0 < sum)
      {
#pragma HLS pipeline
        epsion_over_sum = epsilon;
        epsion_over_sum = epsion_over_sum << 2 * H_QF;
        epsion_over_sum = epsion_over_sum / sum;

        for (i = 0; i < vectorSize; i++)
        {
#pragma HLS pipeline
          temp_data[i] = (temp_data[i] * epsion_over_sum) >> H_QF;
          temp_data[i] = temp_data[i] + (((Data2QF)state_vector[i]) << H_QF);
          temp_data[i] = temp_data[i] * reverse_epsilon;
          state_vector[i] = temp_data[i] >> 2 * H_QF;
        }
      }
    }

    for (i = 0; i < vectorSize; i++)
    {
#pragma HLS pipeline
      channel.data = state_vector[i];
      stream_out.write(channel);
    }
  }

  for (i = 0; i < layerSize; i++)
  {
#pragma HLS pipeline
    channel.data = spike_matrix[i];
    channel.last = (i == layerSize - 1);
    stream_out.write(channel);
  }
}
