# Real Decode Workload

The workload capture came from Qwen1.5-MoE-A2.7B-Chat served by vLLM on A100
with BF16, TP=1, and expert parallelism disabled. Raw traces are intentionally
not included due to size.

`M` is vLLM's total-token launch M. Active-expert M is the actual number of
routed rows consumed by an individual expert GEMM.

## Decode summary

| Metric | Value |
|---|---:|
| Pure Decode layer calls | 21,312 |
| Launch M median / P90 | 8 / 64 |
| Per-active-expert M median / P90 | 3 / 21 |
| Active-expert M <= 8 | 62.00% |
| Active expert ratio median / P90 | 13.33% / 21.67% |
| Padding overhead median / P90 | 300% / 1500% |

## Interpretation

The router and assignment stages run at low M, while routed expert GEMMs see
even smaller and highly irregular per-expert row counts. Padding dominates the
logical routed-token volume for much of Decode. These facts motivated a
workload-driven evaluation rather than optimizing a single synthetic shape.

The captured launch frequencies used for weighted results were:

| M | Calls |
|---:|---:|
| 1 | 3,072 |
| 2 | 3,024 |
| 4 | 3,024 |
| 8 | 3,024 |
| 16 | 3,048 |
| 32 | 3,000 |
| 64 | 2,976 |
