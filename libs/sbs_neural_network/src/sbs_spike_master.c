#include "string.h"

static unsigned int MT19937_flags_ = 0;

typedef enum
{
  INITIALIZED = 1 << 0
} MT19937Flags;

static unsigned int MT19937_initialized ()
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
static void MT19937_sgenrand (unsigned int seed)
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

static unsigned int MT19937_genrand ()
{
    unsigned int y;
    static unsigned int mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if sgenrand() has not been called, */
          MT19937_sgenrand(4357); /* a default initial seed is used   */

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


typedef union
{
  unsigned int    u32;
  float           f32;
} Data32;

#define DATA16_TO_FLOAT32(d)  ((0x30000000 | (((unsigned int)(0xFFFF & (d))) << 12)))
#define DATA8_TO_FLOAT32(d)   ((0x38000000 | (((unsigned int)(0x00FF & (d))) << 19)))

#define FLOAT32_TO_DATA16(d)  (0xFFFF & (((unsigned int)(d)) >> 12))
#define FLOAT32_TO_DATA8(d)   (  0xFF & (((unsigned int)(d)) >> 19))


void sbs_spike_master(unsigned int * spike_matrix_mem,
                      unsigned int * state_matrix_mem,
                      unsigned int rows,
                      unsigned int columns,
                      unsigned int vector_size,
                      unsigned int seed,
                      unsigned int * debug_mem)
{
#pragma HLS INTERFACE m_axi port=spike_matrix_mem offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=state_matrix_mem offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=debug_mem        offset=slave bundle=gmem_debug

#pragma HLS INTERFACE s_axilite port=spike_matrix_mem bundle=control
#pragma HLS INTERFACE s_axilite port=state_matrix_mem bundle=control
#pragma HLS INTERFACE s_axilite port=debug_mem        bundle=control
#pragma HLS INTERFACE s_axilite port=rows             bundle=control
#pragma HLS INTERFACE s_axilite port=columns          bundle=control
#pragma HLS INTERFACE s_axilite port=vector_size      bundle=control
#pragma HLS INTERFACE s_axilite port=seed             bundle=control
#pragma HLS INTERFACE s_axilite port=return           bundle=control

  unsigned int   row;
  unsigned int   column;
  unsigned int   spike_matrix_offset;

  unsigned int   ip_index;
  float          random_s;
  float          sum;
  Data32         h;
  unsigned short spikeID;

  unsigned int  ip_vector[50];
#pragma HLS ARRAY_PARTITION variable=ip_vector complete dim=1
  unsigned int  spike_vector[32 * 32] = { 0 };
  unsigned char spike_vector_index;
  float         random;

  if (!MT19937_initialized ())
  {
    MT19937_sgenrand (seed);
  }

  for (row = 0; row < rows; row++)
  {
    for (column = 0; column < columns; column++)
    {
      spike_matrix_offset = row * columns + column;
      ip_index  = spike_matrix_offset * vector_size;

      memcpy (ip_vector, &state_matrix_mem[ip_index], sizeof(unsigned short) * vector_size);

      random    = MT19937_genrand ();
      random_s  = random / ((float) 0xFFFFFFFF);

      sum       = 0.0f;
      for (spikeID = 0;
           spikeID < vector_size;
           spikeID++)
      {
#pragma HLS PIPELINE
        if (sum < random_s)
        {
          h.u32 = DATA16_TO_FLOAT32(ip_vector[spikeID >> 1] >> ((spikeID & 1) * 16));

          sum += h.f32;

          if (random_s <= sum || (spikeID == vector_size - 1))
          {
            spike_vector[spike_matrix_offset >> 1] |= spikeID << ((spike_matrix_offset & 1) * 16);
          }
        }
      }
    }
  }

  memcpy (spike_matrix_mem, spike_vector, rows * columns * sizeof(unsigned short));
}
