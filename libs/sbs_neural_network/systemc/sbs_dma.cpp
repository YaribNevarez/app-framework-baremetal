#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"
#include "string.h"

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
    mt[0]= seed & 0xffffffff;
    for (mti=1; mti<N; mti++)
        mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;

    MT19937_flags_ |= INITIALIZED;
}

static unsigned int MT19937_rand (unsigned int instance)
{
    unsigned int y;
    static unsigned int mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if sgenrand() has not been called, */
          MT19937_sgenrand(instance, 4357); /* a default initial seed is used   */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return y;
}

typedef unsigned int WeightShift;
typedef unsigned int MT19937;

typedef union
{
  float           f32;
  unsigned int    u32;
} Data32;

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0xFFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (  0xFF & (((unsigned int)(d)) >> 19))

typedef unsigned char   Weight;
typedef unsigned int    Random32;
typedef unsigned short  SpikeID;
typedef unsigned short  Neuron;

typedef struct
{
  unsigned int * data;
  unsigned int rows;
  unsigned int columns;
  unsigned int format_size;
  unsigned int memory_size;
} D2Matrix;

typedef struct
{
  unsigned int * data;
  unsigned int rows;
  unsigned int columns;
  unsigned int length;
  unsigned int entry_size;
  unsigned int data_size;
} D3Matrix;

typedef struct
{
  D3Matrix state_matrix;
  D3Matrix weight_matrix;
  D2Matrix output_spike_matrix;
  D2Matrix input_spike_matrix;

  unsigned int kernel_stride;
  unsigned int kernel_size;
  unsigned int kernel_row_offset;
  unsigned int mt19937;
  float        epsilon;
  unsigned int flags;
} SbsLayerDescriptor;

typedef unsigned int BufferBus;

typedef struct
{
  SbsLayerDescriptor descriptor;
  BufferBus bufferBus[sizeof(SbsLayerDescriptor) / sizeof(BufferBus)];
} SbsLayerDescriptorBuffer;

enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
};

typedef ap_axis<32, 2, 5, 6> StreamChannel;

unsigned int sbs_dma (unsigned int * state_matrix_data,
              unsigned int * weight_matrix_data,
              unsigned int * input_spike_matrix_data,
              unsigned int * output_spike_matrix_data,
              unsigned int * debug,
              unsigned int * buffer,
              hls::stream<StreamChannel> &stream_in,
              hls::stream<StreamChannel> &stream_out,
              unsigned int weight_spikes,
              unsigned int rows,
              unsigned int input_spike_matrix_columns,
              unsigned int input_spike_matrix_rows,
              unsigned int kernel_row_pos,
              unsigned int columns,
              unsigned int vector_size,
              unsigned int kernel_stride,
              unsigned int kernel_size,
              unsigned int layer_weight_shift,
              unsigned int mt19937,
              float epsilon)
{
#pragma HLS INTERFACE m_axi port=state_matrix_data        offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=weight_matrix_data       offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=input_spike_matrix_data  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output_spike_matrix_data offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=debug                    offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=buffer                   offset=slave bundle=gmem

#pragma HLS INTERFACE axis  port=stream_out
#pragma HLS INTERFACE axis  port=stream_in


#pragma HLS INTERFACE s_axilite port=state_matrix_data        bundle=control
#pragma HLS INTERFACE s_axilite port=weight_matrix_data       bundle=control
#pragma HLS INTERFACE s_axilite port=input_spike_matrix_data  bundle=control
#pragma HLS INTERFACE s_axilite port=output_spike_matrix_data bundle=control
#pragma HLS INTERFACE s_axilite port=debug                    bundle=control
#pragma HLS INTERFACE s_axilite port=buffer                   bundle=control

#pragma HLS INTERFACE s_axilite port=weight_spikes              bundle=control
#pragma HLS INTERFACE s_axilite port=rows                       bundle=control
#pragma HLS INTERFACE s_axilite port=input_spike_matrix_columns bundle=control
#pragma HLS INTERFACE s_axilite port=input_spike_matrix_rows    bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_row_pos             bundle=control
#pragma HLS INTERFACE s_axilite port=columns                    bundle=control
#pragma HLS INTERFACE s_axilite port=vector_size                bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_stride              bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_size                bundle=control
#pragma HLS INTERFACE s_axilite port=layer_weight_shift         bundle=control
#pragma HLS INTERFACE s_axilite port=mt19937                    bundle=control
#pragma HLS INTERFACE s_axilite port=epsilon                    bundle=control
#pragma HLS INTERFACE s_axilite port=return                     bundle=control


  static unsigned int input_spike_matrix_buffer[(24 * 24 * sizeof(SpikeID)) / sizeof(unsigned int)] = { 0 };
  static unsigned int weight_matrix_buffer[(1024 * sizeof(Weight)) / sizeof(unsigned int)] = { 0 };
  static unsigned int state_vector_buffer[(1024 * sizeof(Neuron)) / sizeof(unsigned int)] = { 0 };

  static StreamChannel channel;

  unsigned int row;
  SpikeID   spikeID;
  unsigned int column;      /* Column index for navigation on the layer */
  unsigned int kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
  unsigned int kernel_row;        /* Row index for navigation inside kernel */
  unsigned int kernel_column;     /* Column index for navigation inside kernel */
  unsigned int row_column_index;
  unsigned int neuron;
  unsigned char update;
  unsigned int i;
  unsigned int j;
  unsigned int k;
  unsigned int last;
  Data32 data32;
  unsigned int debug_index = 0;
  unsigned int buffer_index = 0;

  if (!MT19937_initialized (mt19937))
  {
    MT19937_sgenrand (mt19937, 666);
  }

  j = input_spike_matrix_rows * input_spike_matrix_columns * sizeof(SpikeID)
      / sizeof(unsigned int);

  i = 0;
  do
  {
    channel = stream_in.read ();
    input_spike_matrix_buffer[i++] = channel.data;
  }
  while (i < j);

  channel.last = 0;

  /* Update begins */
  for (row = 0;
       row < rows;
       row ++,
       kernel_row_pos += kernel_stride)
  {
    for (kernel_column_pos = 0, column = 0;
         column < columns;
         kernel_column_pos += kernel_stride, column ++)
    {
      row_column_index = columns * row + column;

      data32.f32 = ((float) MT19937_rand (mt19937)) / ((float) 0xFFFFFFFF);

      channel.data = data32.u32;

      buffer[buffer_index++] = channel.data;
      stream_out.write (channel);

      memcpy (state_vector_buffer, &state_matrix_data[(vector_size * row_column_index) >> 1], sizeof(unsigned short) * vector_size);

      for (neuron = 0; neuron < vector_size >> 1; neuron ++)
      {
        channel.data = state_vector_buffer[neuron];
        buffer[buffer_index++] = channel.data;
        stream_out.write (channel);
      }

      for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
      {
        for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
        {
          i = (kernel_row_pos + kernel_row) * input_spike_matrix_columns + (kernel_column_pos + kernel_column);
          spikeID = input_spike_matrix_buffer[i >> 1] >> ((i & 1) * 16);
          debug[debug_index++] = spikeID;

          if (layer_weight_shift == COLUMN_SHIFT)
          {
            j = (weight_spikes * kernel_size * kernel_row + weight_spikes * kernel_column + spikeID) * vector_size;
          }
          else
          {
            j = (weight_spikes * kernel_size * kernel_column + weight_spikes * kernel_row + spikeID) * vector_size;
          }

          memcpy(weight_matrix_buffer, &weight_matrix_data[j >> 2], (vector_size + (j & 3)) * sizeof(Weight));

          for (neuron = 0; neuron < vector_size; neuron ++)
          {
            k = neuron + (j & 3);

            if (!(neuron & 3))
            {
              channel.data = 0;
            }

            channel.data |= (0xFF & (weight_matrix_buffer[k >> 2] >> ((k & 3) * 8))) << ((neuron & 3) * 8);

            last = (row == rows - 1)
                && (column == columns - 1)
                && (kernel_row == kernel_size - 1)
                && (kernel_column == kernel_size - 1)
                && (neuron == vector_size - 1);

            if ((neuron & 3) == 3 || last)
            {
              channel.last = last;
              buffer[buffer_index++] = channel.data;
              stream_out.write (channel);
            }
          }
        }
      }
    }
  }
  /* Update ends */
  return debug_index;
}
