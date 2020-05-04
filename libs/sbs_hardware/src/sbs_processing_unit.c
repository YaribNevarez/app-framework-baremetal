/*
 * sbs_hardware.c
 *
 *  Created on: Mar 3rd, 2020
 *      Author: Yarib Nevarez
 */
/***************************** Include Files *********************************/
#include "sbs_processing_unit.h"
#include "sbs_hardware_spike.h"
#include "sbs_hardware_update.h"
#include "dma_hardware_mover.h"
#include "miscellaneous.h"
#include "stdio.h"

#include "mt19937int.h"

/***************** Macros (Inline Functions) Definitions *********************/

/**************************** Type Definitions *******************************/

/************************** Constant Definitions *****************************/

/************************** Variable Definitions *****************************/

/************************** Function Prototypes ******************************/

/************************** Function Definitions******************************/

//#define NUM_ACCELERATOR_INSTANCES  (sizeof(SbSHardwareConfig_list) / sizeof(SbSHardwareConfig))

static SbSUpdateAccelerator **  SbSUpdateAccelerator_list = NULL;
static uint8_t SbSUpdateAccelerator_list_length = 0;

int SbSUpdateAccelerator_getGroupFromList (SbsLayerType layerType,
                                           SbSUpdateAccelerator ** sub_list,
                                           int sub_list_size)
{
  int sub_list_count = 0;
  int i;

  ASSERT (sub_list != NULL);
  ASSERT (0 < sub_list_size);

  ASSERT(SbSUpdateAccelerator_list != NULL);
  for (i = 0; sub_list_count < sub_list_size && i < SbSUpdateAccelerator_list_length;
      i++)
  {
    ASSERT(SbSUpdateAccelerator_list[i] != NULL);
    ASSERT(SbSUpdateAccelerator_list[i]->hardwareConfig != NULL);
    if (SbSUpdateAccelerator_list[i] != NULL
        && SbSUpdateAccelerator_list[i]->hardwareConfig->layerAssign & layerType)
    {
      sub_list[sub_list_count ++] = SbSUpdateAccelerator_list[i];
    }
  }
  return sub_list_count;
}




#define ACCELERATOR_DMA_RESET_TIMEOUT 10000

static void Accelerator_txInterruptHandler (void * data)
{
  DMAHardware * driver  = ((SbSUpdateAccelerator *) data)->hardwareConfig->dmaDriver;
  void *        dma     = ((SbSUpdateAccelerator *) data)->dmaHardware;
  DMAIRQMask irq_status = driver->InterruptGetStatus (dma, MEMORY_TO_HARDWARE);

  driver->InterruptClear (dma, irq_status, MEMORY_TO_HARDWARE);

  if (!(irq_status & DMA_IRQ_ALL)) return;

  if (irq_status & DMA_IRQ_DELAY) return;

  if (irq_status & DMA_IRQ_ERROR)
  {
    int TimeOut;

    ((SbSUpdateAccelerator *) data)->errorFlags |= 0x01;

    driver->Reset (dma);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (driver->ResetIsDone (dma)) break;

    ASSERT(0);
    return;
  }

  if (irq_status & DMA_IRQ_IOC) ((SbSUpdateAccelerator *) data)->txDone = 1;
}

static void Accelerator_rxInterruptHandler (void * data)
{
  SbSUpdateAccelerator *  accelerator = (SbSUpdateAccelerator *) data;
  DMAHardware *           driver      = accelerator->hardwareConfig->dmaDriver;
  void *                  dma         = accelerator->dmaHardware;
  DMAIRQMask              irq_status  = driver->InterruptGetStatus (dma, HARDWARE_TO_MEMORY);

  driver->InterruptClear (dma, irq_status, HARDWARE_TO_MEMORY);

  if (!(irq_status & DMA_IRQ_ALL)) return;

  if (irq_status & DMA_IRQ_DELAY) return;

  if (irq_status & DMA_IRQ_ERROR)
  {
    int TimeOut;

    accelerator->errorFlags |= 0x01;

    driver->Reset (dma);

    for (TimeOut = ACCELERATOR_DMA_RESET_TIMEOUT; 0 < TimeOut; TimeOut--)
      if (driver->ResetIsDone (dma)) break;

    ASSERT(0);
    return;
  }

  if (irq_status & DMA_IRQ_IOC)
  {
//    Xil_DCacheInvalidateRange ((INTPTR) accelerator->rxBuffer,
//                               accelerator->rxBufferSize);

    accelerator->txDone = 1;
    accelerator->rxDone = 1;

    if (accelerator->memory_cmd.cmdID == MEM_CMD_MOVE)
      memcpy (accelerator->memory_cmd.dest,
              accelerator->memory_cmd.src,
              accelerator->memory_cmd.size);
  }
}

static void Accelerator_hardwareInterruptHandler (void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  uint32_t status;

  ASSERT (accelerator != NULL);
  ASSERT (accelerator->hardwareConfig != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver->InterruptGetStatus != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver->InterruptClear != NULL);

#ifdef DEBUG
  if (accelerator->hardwareConfig->hwDriver->Get_debug)
  {
    //int debug = accelerator->hardwareConfig->hwDriver->Get_debug(accelerator->updateHardware);
    //printf ("\nHW interrupt = 0x%X\n", debug);
  }
#endif

  status = accelerator->hardwareConfig->hwDriver->InterruptGetStatus (accelerator->updateHardware);
  accelerator->hardwareConfig->hwDriver->InterruptClear (accelerator->updateHardware, status);
  accelerator->acceleratorReady = status & 1;
}

static unsigned int SbsDMA_debugBuffer[120000 / sizeof(unsigned int)] = { 0 };
static unsigned int SbsDMA_hw_buffer[120000 / sizeof(unsigned int)] = { 0 };
static void Accelerator_hardwareDataMoverInterruptHandler (void * data)
{
  SbSUpdateAccelerator * accelerator = (SbSUpdateAccelerator *) data;
  uint32_t status;

  ASSERT (accelerator != NULL);
  ASSERT (accelerator->hardwareConfig != NULL);
  ASSERT (accelerator->hardwareConfig->hwDataMoverDriver != NULL);
  ASSERT (accelerator->hardwareConfig->hwDataMoverDriver->InterruptGetStatus != NULL);
  ASSERT (accelerator->hardwareConfig->hwDataMoverDriver->InterruptClear != NULL);

#ifdef DEBUG
  if (accelerator->hardwareConfig->hwDataMoverDriver->Get_debug)
  {
    /*
    uint32_t hw_return;
    hw_return = accelerator->hardwareConfig->hwDataMoverDriver->Get_return (accelerator->dataMoverHardware);

    Xil_DCacheInvalidateRange ((INTPTR) SbsDMA_debugBuffer, sizeof(int) * hw_return);
    Xil_DCacheInvalidateRange ((INTPTR) SbsDMA_hw_buffer, sizeof(SbsDMA_hw_buffer));
    */
  }
#endif

  status = accelerator->hardwareConfig->hwDataMoverDriver->InterruptGetStatus (accelerator->dataMoverHardware);
  accelerator->hardwareConfig->hwDataMoverDriver->InterruptClear (accelerator->dataMoverHardware, status);
  accelerator->dataMoverDone = status & 1;
}

int Accelerator_initialize (SbSUpdateAccelerator * accelerator,
                            SbSHardwareConfig * hardware_config)
{
  int                 status;

  ASSERT (accelerator != NULL);
  ASSERT (hardware_config != NULL);

  if (accelerator == NULL || hardware_config == NULL)
    return XST_FAILURE;

  memset (accelerator, 0x00, sizeof(SbSUpdateAccelerator));

  accelerator->hardwareConfig = hardware_config;

  /***************************************************************************/
  accelerator->mt19937 = MT19937_new ();

  ASSERT (accelerator->mt19937 != 0);

  MT19937_initialize (accelerator->mt19937, 666);

  /******************************* DMA initialization ************************/

  accelerator->dmaHardware = hardware_config->dmaDriver->new ();

  ASSERT(accelerator->dmaHardware != NULL);
  if (accelerator->dmaHardware == NULL) return XST_FAILURE;


  status = hardware_config->dmaDriver->Initialize (accelerator->dmaHardware,
                                                   hardware_config->dmaDeviceID);

  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS) return status;

  if (hardware_config->dmaTxIntVecID)
  {
    hardware_config->dmaDriver->InterruptEnable (accelerator->dmaHardware,
                                                 DMA_IRQ_ALL,
                                                 MEMORY_TO_HARDWARE);

    status = hardware_config->dmaDriver->InterruptSetHandler (accelerator->dmaHardware,
                                                              hardware_config->dmaTxIntVecID,
                                                              Accelerator_txInterruptHandler,
                                                              accelerator);
    ASSERT(status != XST_SUCCESS);
    if (status != XST_SUCCESS) return status;
  }

  if (hardware_config->dmaRxIntVecID)
  {
    hardware_config->dmaDriver->InterruptEnable (accelerator->dmaHardware,
                                                 DMA_IRQ_ALL,
                                                 HARDWARE_TO_MEMORY);

    status = hardware_config->dmaDriver->InterruptSetHandler (accelerator->dmaHardware,
                                                              hardware_config->dmaRxIntVecID,
                                                              Accelerator_rxInterruptHandler,
                                                              accelerator);
    ASSERT(status == XST_SUCCESS);
    if (status != XST_SUCCESS) return status;
  }

  /***************************************************************************/
  /**************************** Accelerator initialization *******************/

  accelerator->updateHardware = hardware_config->hwDriver->new();

  ASSERT (accelerator->updateHardware != NULL);

  status = hardware_config->hwDriver->Initialize (accelerator->updateHardware,
                                                  hardware_config->hwDeviceID);
  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS) return XST_FAILURE;

  hardware_config->hwDriver->InterruptGlobalEnable (accelerator->updateHardware);
  hardware_config->hwDriver->InterruptEnable (accelerator->updateHardware, 1);

  status = hardware_config->hwDriver->InterruptSetHandler (accelerator->updateHardware,
                                                           hardware_config->hwIntVecID,
                                                           Accelerator_hardwareInterruptHandler,
                                                           accelerator);
  ASSERT(status == XST_SUCCESS);
  if (status != XST_SUCCESS) return status;

  accelerator->acceleratorReady = 1;
  accelerator->rxDone = 1;
  accelerator->txDone = 1;

  /***************************************************************************/
  /***************************** Data Mover initialization *******************/

  if (hardware_config->hwDataMoverDriver != NULL)
  {
    accelerator->dataMoverHardware = hardware_config->hwDataMoverDriver->new();

    ASSERT (accelerator->dataMoverHardware != NULL);

    status = hardware_config->hwDataMoverDriver->Initialize (accelerator->dataMoverHardware,
                                                             hardware_config->hwDataMoverID);
    ASSERT(status == XST_SUCCESS);
    if (status != XST_SUCCESS) return XST_FAILURE;



    hardware_config->hwDataMoverDriver->InterruptGlobalEnable (accelerator->dataMoverHardware);
    hardware_config->hwDataMoverDriver->InterruptEnable (accelerator->dataMoverHardware, 1);

    status = hardware_config->hwDataMoverDriver->InterruptSetHandler (accelerator->dataMoverHardware,
                                                                      hardware_config->hwDataMoverIntVecID,
                                                                      Accelerator_hardwareDataMoverInterruptHandler,
                                                                      accelerator);
    ASSERT(status == XST_SUCCESS);
    if (status != XST_SUCCESS) return status;

    accelerator->dataMoverDone = 1;
  }

  return XST_SUCCESS;
}

void Accelerator_shutdown(SbSUpdateAccelerator * accelerator)
{
  ASSERT(accelerator != NULL);
  ASSERT(accelerator->hardwareConfig != NULL);

  if ((accelerator != NULL) && (accelerator->hardwareConfig != NULL))
  {
      ARM_GIC_disconnect (accelerator->hardwareConfig->dmaTxIntVecID);

      ARM_GIC_disconnect (accelerator->hardwareConfig->dmaRxIntVecID);

      ARM_GIC_disconnect (accelerator->hardwareConfig->hwIntVecID);
  }
}

SbSUpdateAccelerator * Accelerator_new(SbSHardwareConfig * hardware_config)
{
  SbSUpdateAccelerator * accelerator = NULL;

  ASSERT (hardware_config != NULL);

  if (hardware_config != NULL)
  {
    accelerator = malloc (sizeof(SbSUpdateAccelerator));
    ASSERT (accelerator != NULL);
    if (accelerator != NULL)
    {
      int status = Accelerator_initialize(accelerator, hardware_config);
      ASSERT (status == XST_SUCCESS);

      if (status != XST_SUCCESS)
        free (accelerator);
    }
  }

  return accelerator;
}

void Accelerator_delete (SbSUpdateAccelerator ** accelerator)
{
  ASSERT(accelerator != NULL);
  ASSERT(*accelerator != NULL);

  if ((accelerator != NULL) && (*accelerator != NULL))
  {
    Accelerator_shutdown (*accelerator);

    if ((*accelerator)->updateHardware)
      (*accelerator)->hardwareConfig->hwDriver->delete (&(*accelerator)->updateHardware);

    if ((*accelerator)->dataMoverHardware)
      (*accelerator)->hardwareConfig->hwDataMoverDriver->delete (&(*accelerator)->dataMoverHardware);

    if ((*accelerator)->dmaHardware)
      (*accelerator)->hardwareConfig->dmaDriver->delete (&(*accelerator)->dmaHardware);

    MT19937_delete (&(*accelerator)->mt19937);

    free (*accelerator);
    *accelerator = NULL;
  }
}

#define MASTER_DMA 0

#if MASTER_DMA

#include "xsbs_dma.h"

////////////////////////////////////////////////////////////////////////////////

static unsigned int _MT19937_flags_ = 0;

typedef enum
{
  INITIALIZED = 1 << 0
} _MT19937Flags;

static unsigned int _MT19937_initialized (unsigned int instance)
{
  return _MT19937_flags_ & INITIALIZED;
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
static void _MT19937_sgenrand (unsigned int instance, unsigned int seed)
{
    /* setting initial seeds to mt[N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0]= seed & 0xffffffff;
    for (mti=1; mti<N; mti++)
        mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;

    _MT19937_flags_ |= INITIALIZED;
}

static unsigned int _MT19937_rand (unsigned int instance)
{
    unsigned int y;
    static unsigned int mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if sgenrand() has not been called, */
          _MT19937_sgenrand(instance, 4357); /* a default initial seed is used   */

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

typedef uint8_t   Weight;
typedef uint32_t  Random32;
typedef uint16_t  Neuron;
typedef uint16_t  SpikeID;

typedef union
{
  float           f32;
  unsigned int    u32;
} Data32;

typedef enum
{
  ROW_SHIFT,
  COLUMN_SHIFT
} WeightShift;

typedef struct
{
  int data;
  int last;
} StreamChannel;

typedef struct
{
  StreamChannel (*read) (void);
  void (*write) (StreamChannel);
  unsigned int input_buffer[1];
  unsigned int output_buffer[120000];
  unsigned int input_index;
  unsigned int output_index;
} Stream;

StreamChannel Stream_read (void);
void Stream_write (StreamChannel channel);

Stream Stream_instance = { Stream_read, Stream_write, { 0 }, { 0 }, 0, 0 };

StreamChannel Stream_read (void)
{
  return (StreamChannel ) { Stream_instance.input_buffer[Stream_instance.input_index++], 0 } ;
}

void Stream_write (StreamChannel channel)
{
  /*
  if (SbsDMA_hw_buffer[Stream_instance.output_index]  != channel.data)
  {
    printf ("x");
  }
  */

  Stream_instance.output_buffer[Stream_instance.output_index++] = channel.data;
}

static void sbs_dma (unsigned int * state_matrix_data,
              unsigned int * weight_matrix_data,
              unsigned int * input_spike_matrix_data,
              unsigned int * output_spike_matrix_data,
              Stream stream_in,
              Stream stream_out,
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

#pragma HLS INTERFACE axis  port=stream_out
#pragma HLS INTERFACE axis  port=stream_in


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

  static StreamChannel channel = { 0 };

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

  memcpy(input_spike_matrix_buffer, input_spike_matrix_data, sizeof(SpikeID) * input_spike_matrix_rows * input_spike_matrix_columns);

  if (!_MT19937_initialized (mt19937))
  {
    _MT19937_sgenrand (mt19937, 666);
  }

  channel = stream_in.read ();
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

      data32.f32 = ((float) _MT19937_rand (mt19937)) / ((float) 0xFFFFFFFF);

      channel.data = data32.u32;
      //channel.data = (0xFFFF & row) << 16 | (0xFFFF & column);
      //

      stream_out.write (channel);

      memcpy (state_vector_buffer, &state_matrix_data[(vector_size * row_column_index) >> 1], sizeof(unsigned short) * vector_size);

      for (neuron = 0; neuron < vector_size >> 1; neuron ++)
      {
        channel.data = state_vector_buffer[neuron];
        stream_out.write (channel);
      }

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
              stream_out.write (channel);
            }
          }
        }
      }
    }
  }
  /* Update ends */
}

void sbs_dma_emulator (unsigned int * state_matrix_data,
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
  Stream_instance.input_index = 0;
  Stream_instance.output_index = 0;

  sbs_dma(state_matrix_data,
          weight_matrix_data,
          input_spike_matrix_data,
          output_spike_matrix_data,
          Stream_instance,
          Stream_instance,
          weight_spikes,
          rows,
          input_spike_matrix_columns,
          input_spike_matrix_rows,
          kernel_row_pos,
          columns,
          vector_size,
          kernel_stride,
          kernel_size,
          layer_weight_shift,
          mt19937,
          epsilon);
}

void compare_buffer(SbSUpdateAccelerator * accelerator)
{
  ASSERT(accelerator != NULL);

  for (int i = 0; i < accelerator->txBufferSize / sizeof(int); i++)
    if (((unsigned int *) accelerator->txBuffer)[i] != Stream_instance.output_buffer[i])
    {
      printf ("x");
    }
}
#endif

typedef uint8_t   Weight;
typedef uint32_t  Random32;
typedef uint16_t  Neuron;
typedef uint16_t  SpikeID;

void Accelerator_DMA_setup (SbSUpdateAccelerator * accelerator,
                            unsigned int * state_matrix_data,
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
  if (!accelerator || !accelerator->dataMoverHardware)
    return;

  XSbs_dma_Set_state_matrix_data (accelerator->dataMoverHardware, (unsigned int) state_matrix_data);

  XSbs_dma_Set_weight_matrix_data (accelerator->dataMoverHardware, (unsigned int) weight_matrix_data);

  XSbs_dma_Set_input_spike_matrix_data (accelerator->dataMoverHardware,
                                        (unsigned int) input_spike_matrix_data);

  XSbs_dma_Set_output_spike_matrix_data (accelerator->dataMoverHardware,
                                         (unsigned int) output_spike_matrix_data);

  XSbs_dma_Set_debug (accelerator->dataMoverHardware, (int) SbsDMA_debugBuffer);

  XSbs_dma_Set_buffer_r (accelerator->dataMoverHardware, (int) SbsDMA_hw_buffer);

  XSbs_dma_Set_weight_spikes (accelerator->dataMoverHardware, weight_spikes);

  XSbs_dma_Set_rows (accelerator->dataMoverHardware, rows);

  XSbs_dma_Set_input_spike_matrix_columns (accelerator->dataMoverHardware, input_spike_matrix_columns);

  XSbs_dma_Set_input_spike_matrix_rows (accelerator->dataMoverHardware,
                                        input_spike_matrix_rows);

  XSbs_dma_Set_kernel_row_pos (accelerator->dataMoverHardware, kernel_row_pos);

  XSbs_dma_Set_columns (accelerator->dataMoverHardware, columns);

  XSbs_dma_Set_vector_size (accelerator->dataMoverHardware, vector_size);

  XSbs_dma_Set_kernel_stride (accelerator->dataMoverHardware, kernel_stride);

  XSbs_dma_Set_kernel_size (accelerator->dataMoverHardware, kernel_size);

  XSbs_dma_Set_layer_weight_shift (accelerator->dataMoverHardware, layer_weight_shift);

  XSbs_dma_Set_mt19937 (accelerator->dataMoverHardware, mt19937);

  XSbs_dma_Set_epsilon (accelerator->dataMoverHardware, epsilon);
}

void Accelerator_setup (SbSUpdateAccelerator * accelerator,
                        SbsAcceleratorProfie * profile,
                        AcceleratorMode mode)
{
  ASSERT (accelerator != NULL);
  ASSERT (profile != NULL);

  ASSERT (accelerator->hardwareConfig != NULL);
  ASSERT (accelerator->hardwareConfig->hwDriver != NULL);

  if (accelerator->profile != profile)
  {
    accelerator->profile = profile;

    if (accelerator->hardwareConfig->hwDriver->Set_layerSize)
      accelerator->hardwareConfig->hwDriver->Set_layerSize (
          accelerator->updateHardware, accelerator->profile->layerSize);

    if (accelerator->hardwareConfig->hwDriver->Set_kernelSize)
      accelerator->hardwareConfig->hwDriver->Set_kernelSize (
          accelerator->updateHardware, accelerator->profile->kernelSize);

    if (accelerator->hardwareConfig->hwDriver->Set_vectorSize)
      accelerator->hardwareConfig->hwDriver->Set_vectorSize (
          accelerator->updateHardware, accelerator->profile->vectorSize);

    if (accelerator->hardwareConfig->hwDriver->Set_epsilon)
      accelerator->hardwareConfig->hwDriver->Set_epsilon (
          accelerator->updateHardware, accelerator->profile->epsilon);
  }

  accelerator->mode = mode;
  if (accelerator->hardwareConfig->hwDriver->Set_mode)
    accelerator->hardwareConfig->hwDriver->Set_mode (
        accelerator->updateHardware, accelerator->mode);

  /************************** Rx Setup **************************/
  accelerator->rxBuffer = profile->rxBuffer[mode];
  accelerator->rxBufferSize = profile->rxBufferSize[mode];

  /************************** Tx Setup **************************/
  accelerator->txBuffer = profile->txBuffer[mode];
  accelerator->txBufferSize = profile->txBufferSize[mode];

  ASSERT ((uint32_t)accelerator->hardwareConfig->ddrMem.baseAddress <= (uint32_t)accelerator->rxBuffer);
  ASSERT ((uint32_t)accelerator->rxBuffer + (uint32_t)accelerator->rxBufferSize <= (uint32_t)accelerator->hardwareConfig->ddrMem.highAddress);

  ASSERT ((uint32_t)accelerator->hardwareConfig->ddrMem.baseAddress <= (uint32_t)accelerator->txBuffer);
  ASSERT ((uint32_t)accelerator->txBuffer + (uint32_t)accelerator->txBufferSize <= (uint32_t)accelerator->hardwareConfig->ddrMem.highAddress);

  accelerator->txBufferCurrentPtr = accelerator->txBuffer;

#ifdef DEBUG
  accelerator->txStateCounter = 0;
  accelerator->txWeightCounter = 0;
#endif
}

inline void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         uint32_t * state_vector) __attribute__((always_inline));

inline void Accelerator_giveStateVector (SbSUpdateAccelerator * accelerator,
                                         uint32_t * state_vector)
{
  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->layerSize);
  ASSERT (0 < accelerator->profile->stateBufferSize);
  ASSERT (state_vector != NULL);

  *((float *) accelerator->txBufferCurrentPtr) =
      ((float) MT19937_rand (accelerator->mt19937)) * (1.0 / (float) 0xFFFFFFFF);

  accelerator->txBufferCurrentPtr += sizeof(float);

  memcpy (accelerator->txBufferCurrentPtr,
          state_vector,
          accelerator->profile->stateBufferSize);

  accelerator->txBufferCurrentPtr += accelerator->profile->stateBufferSize;

#ifdef DEBUG
  ASSERT(accelerator->txStateCounter <= accelerator->profile->layerSize);

  accelerator->txStateCounter ++;
#endif
}

inline void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          uint8_t * weight_vector) __attribute__((always_inline));

inline void Accelerator_giveWeightVector (SbSUpdateAccelerator * accelerator,
                                          uint8_t * weight_vector)
{
  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->weightBufferSize);
  ASSERT (0 < accelerator->profile->kernelSize);
  ASSERT (0 < accelerator->profile->layerSize);
  ASSERT (weight_vector != NULL);

#ifdef DEBUG
  ASSERT(accelerator->txWeightCounter <= accelerator->profile->kernelSize * accelerator->profile->layerSize);
#endif

  memcpy (accelerator->txBufferCurrentPtr,
          weight_vector,
          accelerator->profile->weightBufferSize);

  accelerator->txBufferCurrentPtr += accelerator->profile->weightBufferSize;

#ifdef DEBUG
  accelerator->txWeightCounter ++;
#endif
}


int Accelerator_start(SbSUpdateAccelerator * accelerator)
{
  int status;

  ASSERT (accelerator != NULL);
  ASSERT (accelerator->profile != NULL);
  ASSERT (0 < accelerator->profile->stateBufferSize);
  ASSERT (accelerator->mode == SPIKE_MODE || 0 < accelerator->profile->weightBufferSize);
  ASSERT (0 < accelerator->profile->layerSize);

  if (accelerator->dataMoverHardware)
  {
    int flags = 0xA513C85A;
    while (accelerator->dataMoverDone == 0);
    while (accelerator->acceleratorReady == 0);
    while (accelerator->txDone == 0);
    while (accelerator->rxDone == 0);

    accelerator->memory_cmd = accelerator->profile->memory_cmd[accelerator->mode];

    accelerator->acceleratorReady = 0;
    accelerator->hardwareConfig->hwDriver->Start (accelerator->updateHardware);

    accelerator->dataMoverDone = 0;
    accelerator->hardwareConfig->hwDataMoverDriver->Start (accelerator->dataMoverHardware);

    //Xil_DCacheFlushRange ((INTPTR) &flags, sizeof(flags));
    accelerator->txDone = 0;
    status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                           &flags,
                                                           sizeof(flags),
                                                           MEMORY_TO_HARDWARE);
    ASSERT(status == XST_SUCCESS);

    if (status == XST_SUCCESS)
    {
      accelerator->rxDone = 0;
      status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                             accelerator->rxBuffer,
                                                             accelerator->rxBufferSize,
                                                             HARDWARE_TO_MEMORY);

      ASSERT(status == XST_SUCCESS);
    }
  }
  else
  {
    ASSERT ((size_t)accelerator->txBufferCurrentPtr == (size_t)accelerator->txBuffer + accelerator->txBufferSize);

  #ifdef DEBUG
    ASSERT (accelerator->profile->layerSize == accelerator->txStateCounter);
  #endif

    Xil_DCacheFlushRange ((UINTPTR) accelerator->txBuffer, accelerator->txBufferSize);

    while (accelerator->acceleratorReady == 0);
    while (accelerator->txDone == 0);
    while (accelerator->rxDone == 0);

    accelerator->memory_cmd = accelerator->profile->memory_cmd[accelerator->mode];

    accelerator->acceleratorReady = 0;
    accelerator->hardwareConfig->hwDriver->Start (accelerator->updateHardware);


    accelerator->txDone = 0;
    status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                           accelerator->txBuffer,
                                                           accelerator->txBufferSize,
                                                           MEMORY_TO_HARDWARE);
    ASSERT(status == XST_SUCCESS);

    if (status == XST_SUCCESS)
    {
      accelerator->rxDone = 0;
      status = accelerator->hardwareConfig->dmaDriver->Move (accelerator->dmaHardware,
                                                             accelerator->rxBuffer,
                                                             accelerator->rxBufferSize,
                                                             HARDWARE_TO_MEMORY);

      ASSERT(status == XST_SUCCESS);
    }
  }

  return status;
}

/*****************************************************************************/

Result SbsPlatform_initialize (SbSHardwareConfig * hardware_config_list,
                               uint32_t list_length)
{
  int i;
  Result rc;

  rc = ARM_GIC_initialize ();

  ASSERT (rc == OK);

  if (rc != OK)
    return rc;

  if (SbSUpdateAccelerator_list != NULL)
    free (SbSUpdateAccelerator_list);

  SbSUpdateAccelerator_list = malloc(sizeof(SbSUpdateAccelerator *) * list_length);

  ASSERT (SbSUpdateAccelerator_list != NULL);

  rc = (SbSUpdateAccelerator_list != NULL) ? OK : ERROR;

  SbSUpdateAccelerator_list_length = list_length;

  for (i = 0; (rc == OK) && (i < list_length); i++)
  {
    SbSUpdateAccelerator_list[i] = Accelerator_new (&hardware_config_list[i]);

    ASSERT (SbSUpdateAccelerator_list[i] != NULL);

    rc = SbSUpdateAccelerator_list[i] != NULL ? OK : ERROR;
  }

//  if (MT19937_seed)
//    MT19937_sgenrand (MT19937_seed);

  return rc;
}

void SbsPlatform_shutdown (void)
{
  int i;
  ASSERT (SbSUpdateAccelerator_list != NULL);

  if (SbSUpdateAccelerator_list != NULL)
  {
    for (i = 0; i < SbSUpdateAccelerator_list_length; i++)
    {
      Accelerator_delete ((&SbSUpdateAccelerator_list[i]));
    }

    free (SbSUpdateAccelerator_list);
  }
}

/*****************************************************************************/

