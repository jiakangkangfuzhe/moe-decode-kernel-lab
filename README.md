# MoE Decode Kernel Lab

A workload-driven optimization study of vLLM MoE Decode on NVIDIA A100,
including a specialized fused Softmax + TopK4 CUDA kernel.

This repository is a compact engineering record of profiling, hypothesis
testing, kernel implementation, and end-to-end validation. It deliberately
includes negative results: the final Top-k path is 30.25% faster and the
isolated Full MoE is 2.605% faster, but Serving measurements do not show a
stable 2% end-to-end improvement. No larger claim is made.

## Motivation

Mixture-of-Experts inference is easy to benchmark incorrectly. A synthetic
kernel shape can hide routing skew, expert sparsity, padding, allocator costs,
and the interaction between metadata layout and downstream GEMMs. Conversely,
a locally faster preprocessing kernel can make the complete model slower.

This project starts from a real Qwen1.5-MoE serving workload in vLLM rather
than from a hand-selected matrix. It captures Decode routing, extracts the
distribution of launch M and routed rows per active expert, profiles the full
MoE path, and then evaluates narrowly scoped optimization ideas against that
distribution. Kernel results are carried through Full-MoE correctness, CUDA
Graph capture/replay, deterministic generation, and Serving A/B tests.

The goal is not a universal MoE library. It is a reproducible case study in
low-batch GPU optimization: identify the actual cost, estimate the useful
ceiling, implement the smallest justified specialization, and stop when the
end-to-end evidence does not support further complexity.

## Environment

The recorded results use:

- NVIDIA A100 PCIe 40 GB (SM80)
- CUDA 13.0
- PyTorch 2.11.0 with CUDA 13.0
- Triton 3.6.0
- vLLM 0.25.1, upstream commit `752a3a504485790a2e8491cacbb35c137339ad34`
- Qwen1.5-MoE-A2.7B-Chat
- BF16, tensor parallel size 1, expert parallelism disabled

The model and vLLM source are external dependencies. No checkpoints, compiled
extensions, profiler binaries, or raw Serving logs are included.

## Real Decode Workload

The capture contains 21,312 pure Decode layer calls. vLLM's launch M has a
median of 8 and P90 of 64. The actual rows routed to each active expert are
smaller: median 3 and P90 21. Across the active-expert distribution, 62.00% of
shapes have M at most 8. Padding overhead is substantial, with median 300% and
P90 1500%.

| Workload metric | Value |
|---|---:|
| Pure Decode layer calls | 21,312 |
| Launch M median / P90 | 8 / 64 |
| Per-active-expert M median / P90 | 3 / 21 |
| Per-active-expert M <= 8 | 62.00% |
| Padding overhead median / P90 | 300% / 1500% |

These measurements explain why generic large-M intuition is incomplete for
this workload. The router operates on only 60 experts, assignment manipulates
small token sets, and GEMM efficiency depends heavily on grouping and padded
block structure. The exact captured frequency of M=1/2/4/8/16/32/64 is stored
in `results/topk_benchmark.csv` and is used for all weighted latency numbers.
See `docs/workload_analysis.md` for the workload summary.

## Profiling

The complete routed-MoE breakdown showed that no single preprocessing stage
dominates, while GEMM1 remains the largest component:

| Component | Share |
|---|---:|
| Router GEMM | 15.68% |
| Top-k | 12.22% |
| Assignment/Align | 12.29% |
| GEMM1 | 37.84% |
| SwiGLU | 2.54% |
| GEMM2 | 17.06% |
| Finalize | 2.37% |

This breakdown set an important constraint. A Top-k optimization could be
useful, but its Amdahl ceiling is modest. Any preprocessing optimization also
had to preserve the execution quality of GEMM1 and GEMM2.

## Optimization Journey

Four directions were evaluated under explicit GO/KILL criteria:

1. **Triton configuration tuning - KILL.** A search evaluated 4,280
   correctness-valid configurations. The workload-weighted oracle gain was
   only 0.343%, showing that BLOCK_M/N/K, warp count, and pipeline stages were
   not the primary constraint.
2. **Direct small-M GEMM - KILL.** Removing expert grouping and alignment
   regressed the weighted workload by 29.53%. Preprocessing was reduced, but
   GEMM execution quality fell much more.
3. **Assignment/Align - KILL.** An atomic path produced a large isolated local
   speedup, but changed token order inside each expert. GEMM1 latency regressed
   3.584%, and the complete MoE became slower. A controlled ordering-compatible
   variant confirmed the cause.
4. **Softmax + TopK4 fusion - retained.** The non-power-of-two E=60 baseline
   used two kernels and a global FP32 workspace. A single specialized kernel
   removed the intermediate workspace and one launch while matching the
   reference bit-for-bit.

The detailed stop decisions are in `docs/failed_directions.md`. They are kept
separate from `src/` so that the final implementation is unambiguous.

## Final Top-k Optimization

For 60 experts, the baseline vLLM route is:

```text
router logits
  -> moeSoftmax
  -> intermediate FP32 global workspace
  -> moeTopK
  -> topk_weights / topk_ids / source_rows
```

The final path is one CUDA kernel:

```text
load BF16 router logits
  -> reference-order CUB max/sum reduction
  -> FP32 softmax in shared memory
  -> TopK4 selection with reference tie semantics
  -> write weights / ids / source rows
```

The kernel intentionally retains vLLM's 256-thread reduction order. An earlier
warp-reduction prototype was numerically close but not bitwise exact and is not
included in this repository. Preserving the reduction and ArgMax order made
all final weights and IDs exact without giving up the single-launch design.

The runtime dispatch is conservative:

- CUDA on SM80
- BF16 contiguous router logits
- `4 <= E <= 64`
- `top_k=4`
- `0 < M <= 64`
- int32 expert IDs
- standard softmax scoring

Every unsupported case falls back to vLLM's current implementation. The patch
does not alter grouped routing, bias-corrected routing, sigmoid scoring, EP,
prefill, or other GPU architectures.

## Results

| Level | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Top-k path | 51.489 us | 35.912 us | -30.25% |
| Full MoE | 674.533 us | 656.960 us | -2.605% |
| Serving | - | - | no stable >=2% gain |

The Top-k path result is the median GPU latency weighted by the real Decode M
frequency. The Full-MoE result applies the same weighting and changes only the
Top-k implementation. Full-MoE output remained bitwise exact.

The final Serving workload uses prompt length 128, output length 256, and three
independent runs per implementation at each concurrency. Gains are computed
from the median metric across runs; positive latency gain means lower latency.

| C | TPOT median gain | P99 ITL gain | Output tokens/s gain |
|---:|---:|---:|---:|
| 8 | -0.330% | +0.225% | +2.272% |
| 16 | -0.011% | +1.951% | -0.130% |
| 32 | +1.624% | -1.196% | +3.915% |

The Serving results are mixed and are not claimed as a stable end-to-end
speedup. TPOT is effectively flat at C8/C16 and improves 1.624% at C32, while
tail latency and throughput do not move consistently. The final project
decision is `PROJECT_FROZEN_NO_E2E_GAIN`; the Top-k kernel is presented as a
positive sub-optimization rather than a serving-wide win.

## Correctness

The final validation covered the kernel, Full MoE, deterministic generation,
and CUDA Graph execution:

- 49/49 kernel cases passed over M=1/2/4/8/16/32/64, multiple random seeds,
  and explicit ties.
- `topk_ids` are bitwise exact.
- `topk_weights` are bitwise exact; maximum error is 0.
- Full-MoE outputs are bitwise exact at every target M.
- 100/100 generated token sequences are identical at `temperature=0`.
- Default FULL CUDA Graph capture and replay passed.
- Default PIECEWISE CUDA Graph capture and replay passed.

## An Important Negative Result

The align experiment is the most useful caution from this work. An atomic
implementation reduced local assignment latency substantially and produced
metadata with the same expert membership, block counts, padding, and expert
traversal. It was numerically correct. Nevertheless, it changed the order of
`sorted_token_ids` inside each expert.

That layout change made GEMM1's gathers less favorable and increased GEMM1
latency by 3.584%. Reconstructing the original worker-major order removed the
GEMM1 regression, confirming causality, but also removed the preprocessing
advantage. The complete MoE never recovered a positive result.

The general lesson is that layout-equivalent is not necessarily
performance-equivalent. Correctness tests alone cannot detect downstream cache
and gather-locality costs. `docs/align_regression_analysis.md` records the
matched experiment; no align implementation is retained in `src/`.

## Reproduce

Use a local vLLM 0.25.1 checkout and an SM80 CUDA environment. The scripts JIT
build the standalone extension; they never download a model.

```bash
python scripts/benchmark_topk.py \
  --vllm-root /path/to/vllm --output-dir results/reproduced

python scripts/benchmark_full_moe.py \
  --vllm-root /path/to/vllm --output-dir results/reproduced

python -m pytest tests/
```

To exercise the runtime integration, apply
`patches/vllm-0.25.1-topk-runtime.patch`, build the standalone extension with
`python scripts/build_extension.py`, and set `VLLM_MOE_TOPK_LIBRARY` to the
printed library path. The patch documents the opt-in dispatch switch. Raw
traces are not included due to size.

## Repository Layout

- `src/topk/`: final bitwise-correct CUDA kernel and PyTorch registration
- `patches/`: minimal opt-in vLLM 0.25.1 runtime dispatch patch
- `scripts/`: kernel, Full-MoE, and small Serving reproduction tools
- `analysis/`: workload and median-of-runs aggregation tools
- `results/`: small final CSV and Markdown summaries
- `docs/`: architecture, workload, stop decisions, and align analysis
- `tests/`: clean-room CUDA correctness smoke

## Limitations

Results are specific to A100/SM80, Qwen1.5-MoE, BF16, TP=1, EP disabled, and
small-batch Decode. H100, FP8, expert parallelism, tensor parallel sizes above
one, prefill-focused workloads, and other MoE architectures were not
validated. The checked-in patch is an experimental opt-in integration, not an
upstream-ready production claim.

The study also does not claim that Top-k is the next bottleneck for every MoE.
Its value here follows directly from this model's 60-expert, low-M workload and
the baseline's two-launch non-power-of-two path.
