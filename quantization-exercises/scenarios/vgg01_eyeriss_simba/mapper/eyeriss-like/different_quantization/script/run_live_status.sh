#!/bin/bash
# RUN TIMELOOP MAPPER HEURISTIC WITH LIVE STATUS (for 1 thread)

# NOTE: optimized against energy-delay product, if you want to change it, navigate yourself to the mapper folder and change the settings
PROBLEM="../prob/VGG01/vgg01_layer1.yaml"            # choices found in "inputs/prob"
MAPPER="../mapper_live_status/exhaustive_mapper.yaml"
# EYERISS 168 PEs, native eyeriss-like architecture as provided in the presented paper (65nm), only the technology is changed to 45nm
mkdir -p ../outputs_live_status
timeloop-mapper ../arch/eyeriss_like.yaml ../arch/components/*.yaml ../constraints/*.yaml $MAPPER $PROBLEM -o ../outputs_live_status
