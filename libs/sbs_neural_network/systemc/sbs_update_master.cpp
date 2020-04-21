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

static char SbsBaseLayer_updateIP (float * state_vector, float * weight_vector, unsigned int size, float epsilon)
{
  float temp_data[1024];

  unsigned short neuron;

  float sum;
  float reverse_epsilon;
  float epsion_over_sum;

  float h;
  float p;
  float h_p;
  float h_new;

  char update = 0;


  sum = 0.0f;
  for (neuron = 0; neuron < size; neuron++)
  {
    h = state_vector[neuron];
    p = weight_vector[neuron];
    h_p = h * p;

    temp_data[neuron] = h_p;
    sum += h_p;
  }

  if (1e-20 < sum)
  {
    reverse_epsilon = 1.0f / (1.0f + epsilon);
    epsion_over_sum = epsilon / sum;

    for (neuron = 0; neuron < size; neuron++)
    {
      h_p = temp_data[neuron];
      h = state_vector[neuron];

      h_new = reverse_epsilon * (h + h_p * epsion_over_sum);
      state_vector[neuron] = h_new;
    }

    update = 1;
  }

  return update;
}

static unsigned short SbsBaseLayer_generateSpikeIP(float * state_vector_buffer, unsigned short size, MT19937 mt19937)
{
  float random_s = ((float) MT19937_rand (mt19937)) / ((float) 0xFFFFFFFF);
  float sum;
  unsigned short spikeID;

  sum = 0.0f;
  for (spikeID = 0; spikeID < size; spikeID++)
  {
    sum += state_vector_buffer[spikeID];

    if (random_s <= sum)
      return spikeID;
  }

  return size - 1;
}



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

void sbs_update_master (unsigned int * state_matrix_data,
                        unsigned int * weight_matrix_data,
                        unsigned int * input_spike_matrix_data,
                        unsigned int * output_spike_matrix_data,
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


#pragma HLS INTERFACE s_axilite port=state_matrix_data        bundle=control
#pragma HLS INTERFACE s_axilite port=weight_matrix_data       bundle=control
#pragma HLS INTERFACE s_axilite port=input_spike_matrix_data  bundle=control
#pragma HLS INTERFACE s_axilite port=output_spike_matrix_data bundle=control

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
  static unsigned int output_spike_matrix_buffer[(24 * 24 * sizeof(SpikeID)) / sizeof(unsigned int)] = { 0 };
  static unsigned int weight_matrix_buffer[(1024 * sizeof(Weight)) / sizeof(unsigned int)] = { 0 };
  static unsigned int state_vector_buffer[(1024 * sizeof(Neuron)) / sizeof(unsigned int)] = { 0 };

  unsigned int row;
  SpikeID   spikeID;
  unsigned int column;      /* Column index for navigation on the layer */
  unsigned int kernel_column_pos; /* Kernel column position for navigation on the spike matrix */
  unsigned int kernel_row;        /* Row index for navigation inside kernel */
  unsigned int kernel_column;     /* Column index for navigation inside kernel */
  float state_vector[1024];
  float weight_vector[1024];
  unsigned int row_column_index;
  unsigned int neuron;
  unsigned char update;
  unsigned int i;
  unsigned int j;
  Data32 temp;

  memcpy(input_spike_matrix_buffer, input_spike_matrix_data, sizeof(SpikeID) * input_spike_matrix_rows * input_spike_matrix_columns);

  if (!MT19937_initialized (mt19937))
  {
    MT19937_sgenrand (mt19937, 666);
  }

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

      memcpy (state_vector_buffer, &state_matrix_data[(vector_size * row_column_index) >> 1], sizeof(unsigned short) * vector_size);

      for (neuron = 0; neuron < vector_size; neuron ++)
      {
        temp.u32 = DATA16_TO_FLOAT32 (state_vector_buffer[neuron >> 1] >> ((neuron & 1) * 16));
        state_vector[neuron] = temp.f32;
      }

      if ((row_column_index & 1) == 0)
      {
        output_spike_matrix_buffer[row_column_index >> 1] = 0;
      }
      output_spike_matrix_buffer[row_column_index >> 1] |=
          SbsBaseLayer_generateSpikeIP (state_vector, vector_size, mt19937) << (16 * (row_column_index & 1));

      for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
      {
        for (kernel_column = 0; kernel_column < kernel_size; kernel_column++)
        {
          i = (kernel_row_pos + kernel_row) * input_spike_matrix_columns + (kernel_column_pos + kernel_column);
          spikeID = input_spike_matrix_buffer[i >> 1] >> ((i & 1) * 16);

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
            temp.u32 = DATA8_TO_FLOAT32(weight_matrix_buffer[(neuron + (j & 3)) >> 2] >> (((neuron + (j & 3)) & 3) * 8));
            weight_vector[neuron] = temp.f32;
          }

          update = SbsBaseLayer_updateIP (state_vector, weight_vector, vector_size, epsilon);
        }
      }

      if (update)
      {
        for (neuron = 0; neuron < vector_size; neuron ++)
        {
          if (!(neuron & 1))
          {
            state_vector_buffer[neuron >> 1] = 0;
          }

          temp.f32 = state_vector[neuron];

          if ((temp.u32 & 0x30000000) == 0x30000000)
          {
            state_vector_buffer[neuron >> 1] |= FLOAT32_TO_DATA16(temp.u32) << (16 * (neuron & 1));
          }
        }

        memcpy (&state_matrix_data[(vector_size * row_column_index) >> 1], state_vector_buffer, sizeof(Neuron) * vector_size);
      }

    }
  }
  /* Update ends */
  memcpy (output_spike_matrix_data, output_spike_matrix_buffer, sizeof(SpikeID) * rows * columns);
}
