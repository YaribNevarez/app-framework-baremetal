#  Software framework for bare-metal applications
This branch holds a successful implementation of a fully configurable but homogeneous accelerator scheme.

Main HW/SW features:
DMA channel width from 32bit to 1024bit.
Automatically calculates memory padding in SW tx data frame and bitmask and loop unrolling in HW.

Tested on 32, 64, 128bit width.
