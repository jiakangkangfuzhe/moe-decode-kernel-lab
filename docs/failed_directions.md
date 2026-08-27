# Optimization Decisions

This document records directions that were measured and deliberately stopped.
They are useful engineering results, not unbounded follow-up work.

## Config tuning - KILL

- 4,280 correctness-valid BLOCK_M/N/K, warp, and stage configurations.
- Workload-weighted oracle gain: **0.343%**.

The performance bottleneck was not primarily caused by the available Triton
configuration choices. The measured ceiling did not justify runtime dispatch
complexity.

## Direct small-M GEMM - KILL

- Workload-weighted regression: **29.53%**.

Removing expert grouping and assignment reduced GEMM efficiency enough to
outweigh the preprocessing savings. Avoiding one stage is not valuable when it
destroys the execution structure needed by the dominant GEMMs.

## Assignment/Align - KILL

- Best isolated align-path reduction: **70.28%**.
- Runtime GEMM1 latency regression: **3.584%**.
- Complete MoE regressed.

The atomic implementation preserved expert membership, padding, and
expert/block traversal, but changed expert-internal `sorted_token_ids` order.
An ordering-compatible controlled experiment removed the GEMM1 regression,
identifying expert-internal token traversal as the main cause, but also removed
the local align advantage. No end-to-end performance claim is made for this
path.

## Final decision

Top-k fusion was the only positive kernel direction retained. It improved the
isolated Top-k path and Full MoE, but did not produce a stable >=2% Serving
improvement. The project was frozen rather than expanded into another search.
