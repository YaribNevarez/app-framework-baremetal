#  Accelerator Framework
## Design Exploration: Spike-by-Spike Neural Network with Parameterized Floating Point Optimization
This repository contains a heterogeneous collection of specialized hardware accelerators for inference of SbS NN in embedded systems.

| Deployment | Latency (mS / Spike) |
| --- | --- |
| Embedded software | 34.25 |
| Hardware acceleration | 1.66 |

Latency enhancement: 20.62x

## Architecture

### Software
![SbS](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/MNIST/SbS_software_component.png)

### Hardware
![SbS](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/MNIST/hw_architecture.png)

## Heterogeneous accelerator units
| Accelerator unit | Description |
| --- | --- |
| `sbs_spike_unit` | Dedicated hardware to generate spikes from the input layer, this generates **MT19937** random numbers |
| `sbs_accelerator_unit` | Generic SBS update accelerator for layers up to 1024 neurons in inference population |
| `sbs_conv_layer_32` | Convolution layer of 32 neurons in inference population, this stores 1,600 weight coefficients in BRAM |
| `sbs_conv_layer_64` | Convolution layer of 64 neurons in inference population, this stores 51,200 weight coefficients in BRAM |
| `sbs_pooling_layer` | Pooling layers up to 1024 neurons in inference population |

## Deployment
### SbS model
MNIST classification task
![SbS](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/MNIST/sbs_network.png)


### Performance
#### Embedded software
![Performance_SW](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/performance/benchmark_sw.png)

#### Hardware acceleration
![Performance_HW](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/performance/benchmark_hw.png)

#### Accuracy vs spikes
![Accuracy_on_spikes](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/accuracy_vs_spikes/accuracy_on_spikes.png)

#### Accuracy vs noise
![Accuracy_on_noise](https://github.com/YaribNevarez/app-framework-baremetal/blob/hw-accelerator-weight-float4/apps/sbs_app/test/accuracy_vs_spikes/accuracy_on_noise.png)

-Yarib Nevarez
