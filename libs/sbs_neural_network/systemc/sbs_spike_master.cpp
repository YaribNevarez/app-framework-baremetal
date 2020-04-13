#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "hls_stream.h"


typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0xFFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (  0xFF & (((unsigned int)(d)) >> 19))

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

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializing the array with a NONZERO seed */
void sgenrand(unsigned int seed)
{
    /* setting initial seeds to mt[N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0]= seed & 0xffffffff;
    for (mti=1; mti<N; mti++)
        mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;
}

unsigned int genrand()
{
    unsigned int y;
    static unsigned long mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if sgenrand() has not been called, */
            sgenrand(4357); /* a default initial seed is used   */

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

static unsigned char debug_flags = 0;
static unsigned int debug_index = 8;


void sbs_spike_master(unsigned short * spike_matrix_mem,
                      unsigned short * state_matrix_mem,
                      unsigned int rows,
                      unsigned int columns,
                      unsigned int vector_size,
                      unsigned int seed,
                      unsigned int * debug)
{
#pragma HLS INTERFACE m_axi port=spike_matrix_mem offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=state_matrix_mem offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=debug            offset=slave bundle=gmem_debug

#pragma HLS INTERFACE s_axilite port=spike_matrix_mem bundle=control
#pragma HLS INTERFACE s_axilite port=state_matrix_mem bundle=control
#pragma HLS INTERFACE s_axilite port=debug            bundle=control
#pragma HLS INTERFACE s_axilite port=rows             bundle=control
#pragma HLS INTERFACE s_axilite port=columns          bundle=control
#pragma HLS INTERFACE s_axilite port=vector_size      bundle=control
#pragma HLS INTERFACE s_axilite port=seed             bundle=control
#pragma HLS INTERFACE s_axilite port=return           bundle=control

  unsigned int row;
  unsigned int column;
  unsigned int spike_matrix_offset;

  unsigned short * state_vector;
  float random_s;
  float sum;
  Data32 h;
  unsigned short spikeID;
  unsigned short return_spike;
  unsigned int random;

  debug[0] = 1;
  debug[1] = rows;
  debug[2] = columns;
  debug[3] = seed;


  sgenrand(seed);


  debug[4] = 0;
  debug[5] = 0;
  for (row = 0; row < rows; row++)
  {
    debug[4] ++;
    debug[0] = 2;
    for (column = 0; column < columns; column++)
    {
      debug[5]++;
      debug[0] = 3;
      spike_matrix_offset = row * columns + column;

      random = genrand ();
      random_s = ((float) random) / ((float) 0xFFFFFFFF);

      debug[debug_index++] = random;
      h.f32 = random_s;
      debug[debug_index++] = h.u32;

      debug[0] = 4;

      state_vector = &state_matrix_mem[spike_matrix_offset * vector_size];
      return_spike = vector_size - 1;
      sum = 0.0f;
      for (spikeID = 0;
          (spikeID < vector_size) && (return_spike == vector_size - 1);
          spikeID++)
      {
        h.u32 = DATA16_TO_FLOAT32(state_vector[spikeID]);
        debug[debug_index++] = h.u32;
        sum += h.f32;

        debug[0] = 5;

        h.f32 = sum;
        debug[debug_index++] = h.u32;

        if (random_s <= sum)
        {
          h.f32 = random_s;
          debug[debug_index++] = h.u32;
          debug[debug_index++] = spikeID;
          return_spike = spikeID;
          debug[0] = 6;
        }
      }

      spike_matrix_mem[spike_matrix_offset] = return_spike;
    }
  }
}

