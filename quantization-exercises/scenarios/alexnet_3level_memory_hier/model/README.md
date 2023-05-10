# Test scenario for evaluating 1st CONV layer of AlexNet onto a custom HW architecture w and w/o quantization support
Custom 3-level HW architecture with 256KB (2 097 152b) Global on-chip buffer, 8 bits per word and 168 PEs with 64B (512b) local memory Buffer and intmac 8 bit compute unit.

## Problem definition – AlexNet Conv layer 1
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
N=1
C=3
M=96
H=W=227
R=S=11
P=Q=55
Stride=4
Padding=1
<–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>

## Dataflow (mapping) specification (MODEL)
Custom specified mapping.

Specified in the `AlexNet_layer1_3level.map.yaml` file

## Data sparsity
No sparsity specification

## HW sparsity optimization logic
No additional logic

## Scenario run settings
* `native`: native Timeloop behaviour – all data use same bitwidth (specified by the HW memory components)
* `same_quantization`: uniform operands – all operands use same 4-bits datawidth (specified in the input problem specification + internal feasibility checks with HW memory specification)
* `different_quantization`: activations set to 8-bits and weights to 3-bits (should be reflected by memory requirements and in workload evalution)

## Evalution
Inside the `ref-output` folder you can see the reported evaluation stats in the `timeloop-model.stats.txt` file

## Observation
Observe the change in the number of per-level data accesses for the individual scenarios, the utilized capacity and overall energy consumption. Also note that the number of cycles stays the same, meaning that the required number of computations needed to be performed does not change as the total number of elements remains the same. 

Moreover check the op-per byte for the individual memory levels and scenarios.

