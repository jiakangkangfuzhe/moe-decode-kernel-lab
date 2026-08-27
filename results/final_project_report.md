# Final Project Report

## Scope

Qwen1.5-MoE-A2.7B-Chat, NVIDIA A100 PCIe 40 GB / SM80, BF16, TP=1,
expert parallelism disabled, and the vLLM TritonExperts runtime. The final
optimization is restricted to SM80, `4 <= E <= 64`, `top_k=4`, and Decode
`M <= 64`; every other case falls back to vLLM's existing implementation.

## Final results

| Level | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Top-k path | 51.489 us | 35.912 us | -30.25% |
| Full MoE | 674.533 us | 656.960 us | -2.605% |
| Serving | - | - | no stable >=2% gain |

The Top-k and Full-MoE values are weighted by 21,168 captured Decode calls at
`M=1/2/4/8/16/32/64`.

## Correctness and graph safety

- 49/49 kernel cases passed across all target M values, random seeds, and ties.
- `topk_ids` and `topk_weights` are bitwise exact; maximum weight error is 0.
- Full-MoE outputs are bitwise exact for every target M.
- 100/100 generated token sequences matched at `temperature=0`.
- Default FULL and PIECEWISE CUDA Graph capture and replay passed.

## Serving

Decode-heavy, prompt length 128, output length 256, three independent runs per
implementation. Each value below is computed from the median metric across
runs.

| C | TPOT median gain | P99 ITL gain | output tokens/s gain |
|---:|---:|---:|---:|
| 8 | -0.330% | +0.225% | +2.272% |
| 16 | -0.011% | +1.951% | -0.130% |
| 32 | +1.624% | -1.196% | +3.915% |

The Serving results are mixed and are not claimed as a stable end-to-end
speedup.

## Decision

`PROJECT_FROZEN_NO_E2E_GAIN`

The specialized Top-k kernel is retained as a positive sub-optimization. No
further optimization direction was opened after the final validation.
