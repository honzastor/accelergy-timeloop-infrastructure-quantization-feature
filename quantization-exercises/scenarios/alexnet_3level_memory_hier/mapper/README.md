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

## Mapper specification (MAPPER)
Mapper optimizes against:
* `energy and delay`
* `delay and energy`
* `edp`
using random-pruned heuristic and max available threads.

For further details see `mapper.yaml` file

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
Observe how the quantization run settings and the choice of optimized metrics affect the best found mapping in the appropriate results statistics.

Which of the metrics seems like the best choice to optimize against? EDP or other?

