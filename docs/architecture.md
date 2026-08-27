# Architecture

## Measured vLLM MoE path

```text
Qwen2MoeSparseMoeBlock
        |
        v
Router GEMM
        |
        v
Softmax + Top-K
        |
        v
moe_align_block_size
        |
        v
GEMM1
        |
        v
SwiGLU
        |
        v
GEMM2
        |
        v
moe_sum
```

The router is vLLM's default `FusedTopKRouter`. For Qwen1.5-MoE with 60
experts, the baseline non-power-of-two route uses two CUDA kernels and an FP32
workspace:

```text
Before

moeSoftmax
    |
FP32 global workspace
    |
moeTopK
    |
weights / ids / source rows

After

Fused Softmax + TopK4
    |
weights / ids / source rows
```

The final kernel preserves vLLM's 256-thread CUB softmax reduction and ArgMax
tie semantics. The probability row lives in shared memory between softmax and
TopK4. A conservative runtime predicate dispatches only for SM80, CUDA BF16,
`4 <= E <= 64`, `top_k=4`, contiguous inputs, int32 output IDs, and
`0 < M <= 64`. Unsupported inputs use the existing vLLM implementation.

## Why this boundary is narrow

The project validates one serving-relevant point rather than claiming a general
router replacement. It does not cover grouped routing, correction bias,
sigmoid scoring, expert parallelism, other GPU architectures, FP8, or large-M
prefill. This keeps the implementation and its performance claims auditable.
