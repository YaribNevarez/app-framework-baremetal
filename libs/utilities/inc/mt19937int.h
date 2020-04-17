#ifndef LIBS_SBS_MT19937INT_H_
#define LIBS_SBS_MT19937INT_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void * MT19937;

MT19937 MT19937_new ();

void MT19937_delete (MT19937 * instance_ref);

void MT19937_initialize (MT19937 instance, unsigned int seed);

unsigned int MT19937_rand (MT19937 instance);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LIBS_SBS_MT19937INT_H_ */
