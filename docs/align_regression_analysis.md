# Align Regression Analysis

An atomic assignment kernel reduced local preprocessing time, but the complete
MoE became slower. A matched CUDA Graph comparison over 93 Decode model steps
isolated the downstream change:

| Stage | Baseline | Atomic align | Change |
|---|---:|---:|---:|
| Align | 367.620 us | 298.370 us | -18.838% |
| GEMM1 | 3816.046 us | 3952.808 us | +3.584% latency |
| GEMM2 | 2113.318 us | 2156.663 us | +2.051% latency |
| Complete MoE | 6626.562 us | 6744.083 us | +1.773% latency |

## Metadata comparison

For identical routing inputs, `num_tokens_post_pad`, `expert_ids`, per-expert
block counts, expert/block traversal, and padding placement were unchanged.
The difference was the order of valid `sorted_token_ids` inside each expert.
The atomic scatter exposed CUDA scheduling order rather than vLLM's original
worker-major order.

## Controlled experiment

An ordering-compatible variant kept the fast histogram/prefix structure but
reconstructed the baseline token order. GEMM1 recovered from a 3.584%
regression to a 0.640% improvement versus baseline. However, the compatible
scatter made align 18.237% slower than baseline and complete MoE remained
1.289% slower.

The key lesson is that layout-equivalent is not necessarily
performance-equivalent. Numerically correct metadata can change gather
locality and downstream kernel behavior. This path was killed, and its
experimental implementations are intentionally absent from `src/`.
