# Scenario for evaluating 28 CONV layers of MobileNet onto EyerissV2-like HW architecture w and w/o quantization support
This test scenario involves evaluating the twenty eight convolutional (CONV) layers of the MobileNet model on a hardware (HW) architecture similar to Eyeriss V2, both with and without quantized data.

The scenario evaluates original 32-bit float model and also 7 quantized model variants. Each quantized variant uses 8 bits for activations and uniformly either 2-8 bits for weights. According to the model being tested, the hw architecture's word-bits and datatype has also been adequately modified.

The Eyeriss V2 HW architecture is organized as a multi-level hierarchy, with an array of processing elements (PEs) at the lowest level.

At the system level, there is a DRAM which is assumed to be large enough to accommodate all the required data. The technology for this setup is configured to be 45nm, as the open-source estimation plugins offer the highest flexibility at this setting.

The architecture then features a shared on-chip SRAM capable of storing up to 100KB (819 200b) of data.

The Eyeriss-like design is comprised of 168 PEs, each equipped with three local scratchpad (Spads) buffers: the weights Spad, the input feature maps (ifmap) Spad, and the partial sums (psum) Spad. These buffers have storage capacities of 448B (3 584b), 24B (192b), and 48B (384b), respectively.

For testing the quantized models, one scenario modifies the weight Spad size to 0.75 of the original depth and doubles all of the bandwidths.

The ifmap and psum Spads are designed as register files, while the weight Spad is an SRAM memory unit. Each PE performs computations using integer or float values based on the model. The overall word bit width is set according to the scenario – for original data, a 32-bit float data are considered, for the quantized models we modify the architecture to use either 8-bit int or 16-bit int word-data.

You can find more details about the architecture description in the `arch` folder of the individual test scenarios.

## Problem definition – MobileNet Conv layers
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
MobileNet Conv 1 layer definition (for other layer shapes see the `prob` folder)
C=3
N=M=1
H=W=224
R=S=3
P=Q=112
Stride=2
Padding=1
<–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>

## Mapper specification (MAPPER)
Mapper optimizes against:
* `energy and delay`
* `delay and energy`
* `edp`
* `energy and last-level-accesses`
* `delay and last-level-accesses`
* `last-level-accesses and energy`
* `last-level-accesses and delay`

Using 4 different choice of heuristics: `random-pruned`, `linear-pruned`, `hybrid` and `exhaustive` using 1 or max available threads.

For further details navigate to the `mapper` folder

## Data sparsity
No sparsity specification

## HW sparsity optimization logic
No additional logic

## Scenario run settings
* `original_float_data`: original model, considering all data tensors are 32-bit floats
* `quantized_data`: 7 different model quantization settings, each using 8-bit input and output fmaps and between 2-8 bit weights (uniform setting across all layers)

## Evalution
Inside the `ref-output` folder you can see the reported evaluation stats in the `timeloop-model.stats.txt` and `timeloop-model.stats.csv` files.

## Observation
Observe the change in the reported stats and best found mappings across individual scenarios.

Observe how does the choice of bitwith correlate with the HW design, do some quantized models perform better or worse on certain architecture modification?

