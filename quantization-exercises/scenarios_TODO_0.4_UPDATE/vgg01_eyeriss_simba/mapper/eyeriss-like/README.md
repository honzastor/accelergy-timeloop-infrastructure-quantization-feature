# Test scenario for evaluating 8 CONV layers of VGG01 onto EyerissV2-like HW architecture w and w/o quantization support
This test scenario involves evaluating the eight convolutional (CONV) layers of the VGG01 model on a hardware (HW) architecture similar to Eyeriss V2, both with and without quantized data.

NOTE: The VGG01 model used is based on a paper: [Shafiee, ISCA 2016](https://ieeexplore.ieee.org/document/7551379)

The Eyeriss V2 HW architecture is organized as a multi-level hierarchy, with an array of processing elements (PEs) at the lowest level.

At the system level, there is a DRAM which is assumed to be large enough to accommodate all the required data. The technology for this setup is configured to be 45nm, as the open-source estimation plugins offer the highest flexibility at this setting.

The architecture then features a shared on-chip SRAM capable of storing up to 100KB (819 200b) of data.

The Eyeriss-like design is comprised of 168 PEs, each equipped with three local scratchpad (Spads) buffers: the weights Spad, the input feature map (ifmap) Spad, and the partial sum (psum) Spad. These buffers have storage capacities of 448B (3 584b), 24B (192b), and 48B (384b), respectively.

The ifmap and psum Spads are designed as register files, while the weight Spad is an SRAM memory unit. Each PE performs computations using integer values, and the overall word bit width is set to 16 bits. 

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