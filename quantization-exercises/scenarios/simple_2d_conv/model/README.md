# Simple test scenario for evaluating basic CONV layer onto a simple HW architecture w and w/o quantization support
Simple architecture with 1 on-chip memory Buffer storing 512 bits (64B) of data in total, 8 bits per word and intmac 8 bit compute units.

## Problem definition – 2D Convolution
<–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>
          Ifmap               Weights                Ofmap
   C                        C                 M
    ⤢                        ⤢                 ⤢
     ┌──────────────┐          ┌────┐            ┌─────────┐
     │              │        R │    │            │         │
     │              │          └────┘          P │         │
   H │              │   *        S       =       │         │
     │              │                            └─────────┘
     │              │        ↑                        Q
  ↑  └──────────────┘      M │   ⋮
N │         W                ↓
  │         ⋮
  ↓
<–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>
C=N=M=1
H=W=6
R=S=3
P=Q=4
Stride=1
Padding=0
<–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>

## Dataflow (mapping) specification
Weight stationary dataflow (weights change less frequently)

Specified in the `conv-1level.map.yaml` file

## Data sparsity
No sparsity specification

## HW sparsity optimization logic
No additional logic

## Scenario run settings
* `native`: native Timeloop behaviour – all data use same bitwidth (specified by the HW memory components)
* `same_quantization`: uniform operands – all operands use same 4-bits datawidth (specified in the input problem specification + internal feasibility checks with HW memory specification)
* `different_quantization`: activations set to 8-bits and weights to 3-bits (should be reflected by memory requirements and in workload evalution)

## Evalution
Inside the `ref-output` folder you can see the reported evaluation stats in the `timeloop-model.stats.txt file`

## Observation
Observe the change in the number of per-level data accesses for the individual scenarios, the utilized capacity and overall energy consumption. Also note that the number of cycles stays the same, meaning that the required number of computations needed to be performed does not change as the total number of elements remains the same. 
You can also observe the effects that the quantized data tensor has on the wasted memory bits and the amount of data fragmentation.

