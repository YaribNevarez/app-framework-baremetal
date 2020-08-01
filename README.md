#  Accelerator Framework of Spike-by-Spike Neural Networks in Embedded Systems (Xilinx FPGA)
This repository contains a heterogeneous collection of specialized hardware accelerators for inference of SbS NN in embedded systems. Ths implementation achieves 2.3 latency for inference in cyclic operation.

## Functional features
The following is a list of the specialized hardware:

| Processing unit | Description |
| --- | --- |
| `sbs_spike_unit` | Dedicated hardware to generate spikes from the input layer, this generates **MT19937** random numbers |
| `sbs_accelerator_unit` | Generic SBS update accelerator for layers up to 1024 neurons in inference population |
| `sbs_conv_layer_32` | Convolution layer of 32 neurons in inference population, this stores 1,600 weight coefficients in BRAM |
| `sbs_conv_layer_64` | Convolution layer of 64 neurons in inference population, this stores 51,200 weight coefficients in BRAM |
| `sbs_pooling_layer` | Pooling layers up to 1024 neurons in inference population |

## Tooling features
* Performance measurement instrumented with `Event` class
* Log2 histogram generation in `Multivectr` class

## Bug fixing
* DATA16_TO_FLOAT32(), DATA08_TO_FLOAT32(), FLOAT32_TO_DATA16(), FLOAT32_TO_DATA08()

## Known issues
* Input layer tx buffer in spike accelerator is not being updated causing misclassification after the first loaded input
