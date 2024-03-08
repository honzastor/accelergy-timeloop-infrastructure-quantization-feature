#!/bin/bash
# RUN TIMELOOP MAPPER HEURISTIC WITH LIVE STATUS (for 1 thread)

# NOTE: optimized against energy-delay product, if you want to change it, navigate yourself to the mapper folder and change the settings
PROBLEM="../prob/VGG01/vgg01_layer1.yaml"            # choices found in "inputs/prob"
MAPPER="../mapper_live_status/exhaustive_mapper.yaml"
# Simba-like (only 1 chip from original simba chiplet structure, architecture similar to NVDLA), configured on 45nm technology using 16-bit data and word widths, the architecture contains 16 PEs
mkdir -p ../outputs_live_status
timeloop-mapper ../arch/simba_like.yaml ../arch/components/*.yaml ../constraints/*.yaml $MAPPER $PROBLEM -o ../outputs_live_status
