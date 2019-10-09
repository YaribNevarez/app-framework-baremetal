#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

typedef union
{
	unsigned int u32;
	float        f32;
} FloatToInt;

#define MAX_LAYER_SIZE      (28*28)
#define MAX_KERNEL_SIZE     (5*5)
#define MAX_VECTOR_SIZE     (1024)

typedef ap_axis<32,2,5,6> StreamChannel;

void sbs_update(StreamChannel stream_in[MAX_LAYER_SIZE * (MAX_KERNEL_SIZE + 1)* MAX_VECTOR_SIZE],
				StreamChannel stream_out[MAX_LAYER_SIZE * MAX_VECTOR_SIZE],
				int layerSize,
				int kernelSize,
				int vectorSize,
				float epsilon)
{
#pragma HLS INTERFACE axis      port=stream_in
#pragma HLS INTERFACE axis      port=stream_out
#pragma HLS INTERFACE s_axilite port=layerSize   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=kernelSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=vectorSize  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=epsilon     bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return      bundle=CRTL_BUS

  static int ip_index;
  static int i;
  static int j = 0;
  static int batch;
  static StreamChannel last;
  static float state_vector[MAX_VECTOR_SIZE];
  static float temp_data[MAX_VECTOR_SIZE];
  static float epsion_over_sum;

  static FloatToInt float_to_int;

  float reverse_epsilon = 1.0f / (1.0f + epsilon);
  static float sum;

  for (ip_index = 0; ip_index < layerSize; ip_index ++)
  {
#pragma HLS pipeline
	  for (i = 0; i < vectorSize; i++)
	  {
	#pragma HLS pipeline
		  if (i == 0)
		  {
			  last = stream_in[i];
			  float_to_int.u32 = last.data;
		  }
		  else
		  {
			  float_to_int.u32 = stream_in[i].data;
		  }

		state_vector[i] = float_to_int.f32;
	  }

	  for (batch = 0; batch < kernelSize; batch++)
	  {
	#pragma HLS pipeline
		j = vectorSize * (1 + batch);
		sum = 0.0f;
		for (i = 0; i < vectorSize; i++)
		{
	#pragma HLS pipeline
		  float_to_int.u32 = stream_in[i + j].data;
		  temp_data[i] = state_vector[i] * float_to_int.f32;
		  sum += temp_data[i];
		}

		epsion_over_sum = epsilon / sum;

		if (sum > 1e-20) // TODO: DEFINE constant
		  for (i = 0; i < vectorSize; i++)
		  {
	#pragma HLS pipeline
			state_vector[i] = reverse_epsilon * (state_vector[i] + temp_data[i] * epsion_over_sum);
		  }
	  }

	  for (i = 0; i < vectorSize; i++)
	  {
	#pragma HLS pipeline
		float_to_int.f32 = state_vector[i];

		last.data = float_to_int.u32;
		last.last = (i == vectorSize - 1) && (ip_index == layerSize - 1);

		stream_out[i] = last;
	  }
  }
}

