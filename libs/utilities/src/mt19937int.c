#include "stdio.h"
#include "mt19937int.h"

#include "xmt19937_rand.h"
#include "xparameters.h"

static XMt19937_rand Mt19937_rand;

void MT19937_sgenrand (unsigned int seed)
{
  XMt19937_rand_Initialize (&Mt19937_rand, XPAR_MT19937_RAND_0_DEVICE_ID);

  XMt19937_rand_Set_seed (&Mt19937_rand, seed);

  XMt19937_rand_Start (&Mt19937_rand);
}

unsigned int MT19937_genrand ()
{
  unsigned int ret;
  while (!XMt19937_rand_IsDone (&Mt19937_rand));

  ret = XMt19937_rand_Get_return (&Mt19937_rand);

  XMt19937_rand_Start (&Mt19937_rand);

  return ret;
}

