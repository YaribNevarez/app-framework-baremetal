#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"

#include <stdio.h>
#include <string.h>

#define CHANNEL_SIZE 1024
#define CACHE_ARRAY_LENGTH 1024

typedef ap_axis<CHANNEL_SIZE, 2, 5, 6>    Channel;

typedef enum
{
  MASTER_DIRECT,
  MASTER_DIRECT_PIPELINE,
  MASTER_CACHED,
  MASTER_CACHED_PIPELINED,
  MASTER_CACHED_BURST,
  MASTER_STORE_BURST,
  MASTER_FLUSH_BURST,
  MASTER_STORE,
  MASTER_FLUSH,
  MASTER_STORE_PIPELINED,
  MASTER_FLUSH_PIPELINED,
  STREAM_DIRECT,
  STREAM_DIRECT_PIPELINED,
  STREAM_CACHED,
  STREAM_CACHED_PIPELINED,
  STREAM_STORE,
  STREAM_FLUSH,
  STREAM_STORE_PIPELINED,
  STREAM_FLUSH_PIPELINED
} TestCase;

unsigned int test_module (unsigned int test_case,
                          unsigned int buffer_length,
                          ap_uint<CHANNEL_SIZE> *   master_in,
                          ap_uint<CHANNEL_SIZE> *   master_out,
                          hls::stream<Channel> &   stream_in,
                          hls::stream<Channel> &   stream_out)
{
#pragma HLS INTERFACE m_axi  port=master_in    offset=slave   bundle=MASTER_BUS
#pragma HLS INTERFACE m_axi  port=master_out   offset=slave   bundle=MASTER_BUS

#pragma HLS INTERFACE axis       port=stream_in
#pragma HLS INTERFACE axis       port=stream_out

#pragma HLS INTERFACE s_axilite  port=test_case   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite  port=buffer_length   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite  port=master_in   bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite  port=master_out  bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite  port=return      bundle=CRTL_BUS

  unsigned int result = 0;
  Channel      channel_in;
  Channel      channel_out;

  channel_out.keep = -1;
  channel_out.strb = -1;

  static ap_uint < CHANNEL_SIZE > cache_array[CACHE_ARRAY_LENGTH];
//#pragma HLS array_partition variable=cache_array block factor=1 dim=0

  switch (test_case)
  {
    case MASTER_DIRECT:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        master_out[i] = master_in[i];
      }
      break;

    case MASTER_DIRECT_PIPELINE:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        master_out[i] = master_in[i];
      }
      break;

    case MASTER_CACHED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        cache_array[i] = master_in[i];
      }

      for (unsigned int i = 0; i < buffer_length; i ++)
      {
        master_out[i] = cache_array[i];
      }
      break;

    case MASTER_CACHED_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        cache_array[i] = master_in[i];
      }

      for (unsigned int i = 0; i < buffer_length; i ++)
      {
#pragma HLS pipeline
        master_out[i] = cache_array[i];
      }
      break;

    case MASTER_CACHED_BURST:
      memcpy (cache_array, master_in, sizeof(ap_uint< CHANNEL_SIZE> ) * buffer_length);
      memcpy (master_out, cache_array, sizeof(ap_uint< CHANNEL_SIZE> ) * buffer_length);
      break;

    case MASTER_STORE_BURST:
      memcpy (cache_array, master_in, sizeof(ap_uint< CHANNEL_SIZE> ) * buffer_length);
      break;

    case MASTER_FLUSH_BURST:
      memcpy (master_out, cache_array, sizeof(ap_uint< CHANNEL_SIZE> ) * buffer_length);
      break;

    case MASTER_STORE:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        cache_array[i] = master_in[i];
      }
      break;

    case MASTER_FLUSH:
      for (unsigned int i = 0; i < buffer_length; i ++)
      {
        master_out[i] = cache_array[i];
      }
      break;

    case MASTER_STORE_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        cache_array[i] = master_in[i];
      }
      break;

    case MASTER_FLUSH_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i ++)
      {
#pragma HLS pipeline
        master_out[i] = cache_array[i];
      }
      break;

    case STREAM_DIRECT:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        stream_in.read (channel_in);
        stream_out.write (channel_in);
      }
      break;

    case STREAM_DIRECT_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        stream_in.read (channel_in);
        stream_out.write (channel_in);
      }
      break;

    case STREAM_CACHED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        stream_in.read (channel_in);
        cache_array[i] = channel_in.data;
      }

      for (unsigned int i = 0; i < buffer_length; i++)
      {
        channel_out.data = cache_array[i];
        channel_out.last = buffer_length - 1 == i;
        stream_out.write (channel_out);
      }
      break;

    case STREAM_CACHED_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        stream_in.read (channel_in);
        cache_array[i] = channel_in.data;
      }

      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        channel_out.data = cache_array[i];
        channel_out.last = buffer_length - 1 == i;
        stream_out.write (channel_out);
      }
      break;

    case STREAM_STORE:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        stream_in.read (channel_in);
        cache_array[i] = channel_in.data;
      }
      break;

    case STREAM_FLUSH:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
        channel_out.data = cache_array[i];
        channel_out.last = buffer_length - 1 == i;
        stream_out.write (channel_out);
      }
      break;

    case STREAM_STORE_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        stream_in.read (channel_in);
        cache_array[i] = channel_in.data;
      }
      break;

    case STREAM_FLUSH_PIPELINED:
      for (unsigned int i = 0; i < buffer_length; i++)
      {
#pragma HLS pipeline
        channel_out.data = cache_array[i];
        channel_out.last = buffer_length - 1 == i;
        stream_out.write (channel_out);
      }
      break;

    default:
      result = -1;
  }

  return result;
}
