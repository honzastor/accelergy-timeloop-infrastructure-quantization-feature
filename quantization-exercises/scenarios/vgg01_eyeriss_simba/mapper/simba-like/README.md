# Test scenario for evaluating 8 CONV layers of VGG01 onto Simba-like HW architecture w and w/o quantization support
This test scenario involves evaluating the eight convolutional (CONV) layers of the VGG01 model on a hardware (HW) architecture similar to to a single chip of Simba chiplet (similar design do NVDLA), both with and without quantized data.

The Simba HW architecture is organized as a multi-level hierarchy, with an array of processing elements (PEs) at the lowest level.

At the system level, there is a DRAM which is assumed to be large enough to accommodate all the required data. The technology for this setup is configured to be 45nm, as the open-source estimation plugins offer the highest flexibility at this setting.

The architecture then features a shared on-chip SRAM Global Buffer capable of storing up to 64KB (524 288b) of data.

The Simba-like design is comprised of 16 PEs, each equipped with 1 smartbuffer for input activations, capable of storing 64KB (524 288b) data; 4 smartbuffers for storing weights, each of which has capacity of 8192 words (depends on word-size); 4 smartbuffers for accumulating the partial sums, each with a storage size of 128 words (depends on word-size); 16 registers for feeding the compute units, each with size of 1 word (depends on word-size).

Each PE performs computations using integer values, and the overall word bit width is set to 16 bits.


## Problem definition – VGG01 Conv layers
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
VGG01 Conv 1 layer definition (for other layer shapes see the `prob` folder)
N=1
C=3
M=64
H=W=224
R=S=3
P=Q=224
Stride=1
Padding=1
<–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>

## Mapper specification (MAPPER)
Mapper optimizes against:
* `energy and delay`
* `delay and energy`
* `edp`

Using 4 different choice of heuristics: `random-pruned`, `linear-pruned`, `hybrid` and `exhaustive` using 1 or max available threads.

For further details navigate to the `mapper` folder

## Data sparsity
No sparsity specification

## HW sparsity optimization logic
No additional logic

## Scenario run settings
* `native`: native Timeloop behaviour – all data use same bitwidth (specified by the HW memory components)
* `same_quantization`: uniform operands – all operands use same 4-bits datawidth (specified in the input problem specification + internal feasibility checks with HW memory specification)
* `different_quantization`: activations set to 8-bits and weights to 3-bits (should be reflected by memory requirements and in workload evalution)

## Evalution
Inside the `ref-output` folder you can see the reported evaluation stats in the `timeloop-model.stats.txt` and `timeloop-model.stats.csv` files.

## Observation 
Observe how the quantization run settings and the choice of optimized metrics and heuristic affect the best found mapping in the appropriate results statistics.

Compare the results with the other HW architecture.