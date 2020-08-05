/*
 * sbs_nn.c
 *
 *  Created on: Sep 7, 2019
 *      Author: Yarib Nevarez
 */


//#define DEBUG //  -g3 -O0 -DDEBUG

#include "sbs_neural_network.h"
#include "miscellaneous.h"
#include "sleep.h"

/*****************************************************************************/

#pragma pack(push)  /* push current alignment to stack */
#pragma pack(1)     /* set alignment to 1 byte boundary */

typedef struct
{
  SbsNetwork        vtbl;
  uint8_t           size;
  SbsBaseLayer **   layer_array;
  uint8_t           input_label;
  uint8_t           inferred_output;
  Event *           event;
} SbsBaseNetwork;

#pragma pack(pop)   /* restore original alignment from stack */

static SbsNetwork * SbsBaseNetwork_new (void)
{
  SbsBaseNetwork * network = NULL;

  network = malloc (sizeof(SbsBaseNetwork));

  ASSERT(network != NULL);

  if (network != NULL)
  {
    memset (network, 0x0, sizeof(SbsBaseNetwork));
    network->vtbl = _SbsNetwork;
    network->input_label = (uint8_t) -1;
    network->inferred_output = (uint8_t) -1;
    network->event = Event_new (NULL, "SbS_Network");
  }

  ASSERT(network->size == 0);
  ASSERT(network->layer_array == NULL);

  return (SbsNetwork *) network;
}

static void SbsBaseNetwork_delete (SbsNetwork ** network_ptr)
{
  ASSERT(network_ptr != NULL);
  ASSERT(*network_ptr != NULL);

  if ((network_ptr != NULL) && (*network_ptr != NULL))
  {
    SbsBaseNetwork ** network = (SbsBaseNetwork **) network_ptr;
    while (0 < (*network)->size)
      SbsBaseLayer_delete ((SbsLayer **)&(*network)->layer_array[--((*network)->size)]);

    if ((*network)->event != NULL)
      Event_delete (&(*network)->event);

    free(*network);
    *network = NULL;
  }
}

static void SbsBaseNetwork_giveLayer (SbsNetwork * network_ptr, SbsLayer * layer)
{
  ASSERT(network_ptr != NULL);
  ASSERT(layer != NULL);

  if ((network_ptr != NULL) && (layer != NULL))
  {
    SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
    SbsBaseLayer ** layer_array = network->layer_array;
    uint8_t size = network->size;

    ASSERT(size < 0xFF);

    layer_array = realloc (layer_array, (size + 1) * sizeof(SbsBaseLayer *));

    ASSERT(layer_array != NULL);

    if (layer_array != NULL)
    {
      layer_array[size] = (SbsBaseLayer *) layer;

      SbsBaseLayer_setParentEvent ((SbsBaseLayer *) layer, network->event);

      network->layer_array = layer_array;
      network->size++;
    }
  }
}

static Result SbsBaseNetwork_loadInput (SbsNetwork * network_ptr,
                                        char * file_name)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  Result result = ERROR;

  ASSERT(network != NULL);
  ASSERT(1 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(*network->layer_array != NULL);
  ASSERT(file_name != NULL);

  if ((network != NULL)
      && (1 <= network->size)
      && (network->layer_array != NULL)
      && (*network->layer_array != NULL)
      && (file_name != NULL))
  {
    result = SbsBaseLayer_loadInput (network->layer_array[0],
                                     file_name,
                                     &network->input_label);
  }

  return result;
}

static void SbsBaseNetwork_updateInferredOutput(SbsBaseNetwork * network)
{
  ASSERT (network != NULL);

  if (network != NULL)
  {
    float max_value = 0;
    SbsBaseLayer * output_layer = network->layer_array[network->size - 1];
    float output_state_vector[10] = {0};
    uint16_t output_vector_size = 10;
    float h;
    int i;

    SbsBaseLayer_getOutputVector (output_layer,
                                  output_state_vector,
                                  output_vector_size);

    ASSERT (output_state_vector != NULL);
    ASSERT (0 < output_vector_size);

    for (i = 0; i < output_vector_size; i++)
    {
      h = output_state_vector[i];

      if (max_value < h)
      {
        network->inferred_output = i;
        max_value = h;
      }
    }
  }
}

static void SbsBaseNetwork_updateCycle(SbsNetwork * network_ptr, uint16_t cycles)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;

  ASSERT(network != NULL);
  ASSERT(3 <= network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(0 < cycles);

  if ((network != NULL) && (3 <= network->size)
      && (network->layer_array != NULL) && (cycles != 0))
  {
    int i;

    /* Initialize all layers except the input-layer */
    for (i = 0; i < network->size; i++)
    {
      ASSERT(network->layer_array[i] != NULL);
      SbsBaseLayer_initialize (network->layer_array[i]);
      SbsBaseLayer_cacheFlush (network->layer_array[i]);
      SbsBaseLayer_initializeProcessingUnit (network->layer_array[i]);
    }

    /************************ Begins Update cycle ****************************/
    while (cycles--)
    {
      Event_start (network->event);
      for (i = 0; i <= network->size - 1; i++)
      {
        network->layer_array[i]->process (network->layer_array[i],
                                          i ? network->layer_array[i - 1] : NULL);
      }
      Event_stop (network->event);

    }
    usleep (10000);
    Event_print (network->event);

    /************************ Ends Update cycle ******************************/

    /************************ Get inferred output **************************/
    SbsBaseNetwork_updateInferredOutput (network);
  }
}

static uint8_t SbsBaseNetwork_getInferredOutput(SbsNetwork * network)
{
  uint8_t inferred_output = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    inferred_output = ((SbsBaseNetwork *) network)->inferred_output;
  }

  return inferred_output;
}

static uint8_t SbsBaseNetwork_getInputLabel(SbsNetwork * network)
{
  uint8_t input_label = (uint8_t)-1;

  ASSERT(network != NULL);
  if (network != NULL)
  {
    input_label = ((SbsBaseNetwork *) network)->input_label;
  }

  return input_label;
}

static void SbsBaseNetwork_getOutputVector(SbsNetwork * network_ptr,
                                           float * output_vector,
                                           uint16_t output_vector_size)
{
  SbsBaseNetwork * network = (SbsBaseNetwork *) network_ptr;
  ASSERT(network != NULL);
  ASSERT(0 < network->size);
  ASSERT(network->layer_array != NULL);
  ASSERT(network->layer_array[network->size - 1] != NULL);

  ASSERT(output_vector != NULL);
  ASSERT(output_vector_size != 0);

  if ((network != NULL)
      && (0 < network->size)
      && (network->layer_array != NULL)
      && (network->layer_array[network->size - 1] != NULL)
      && (output_vector != NULL)
      && (output_vector_size != 0))
  {
    SbsBaseLayer_getOutputVector (network->layer_array[network->size - 1],
                                  output_vector,
                                  output_vector_size);
  }
}

static void SbsBaseNetwork_printStatistics (SbsNetwork * network)
{
  usleep (10000);
  Event_print (((SbsBaseNetwork*) network)->event);
}

/*****************************************************************************/

SbsNetwork _SbsNetwork = {SbsBaseNetwork_new,
                          SbsBaseNetwork_delete,
                          SbsBaseNetwork_giveLayer,
                          SbsBaseNetwork_loadInput,
                          SbsBaseNetwork_updateCycle,
                          SbsBaseNetwork_getInferredOutput,
                          SbsBaseNetwork_getInputLabel,
                          SbsBaseNetwork_getOutputVector,
                          SbsBaseNetwork_printStatistics};

SbsNew sbs_new = {SbsBaseNetwork_new,
                  SbsBaseLayer_new,
                  SbsWeightMatrix_new,
                  SbsInputLayer_new,
                  SbsConvolutionLayer_new,
                  SbsPoolingLayer_new,
                  SbsFullyConnectedLayer_new,
                  SbsOutputLayer_new};


/*****************************************************************************/
