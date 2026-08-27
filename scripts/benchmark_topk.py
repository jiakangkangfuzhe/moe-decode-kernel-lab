#!/usr/bin/env python3
"""Benchmark the vLLM Top-k path against the final SM80 specialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DECODE_FREQUENCIES,
    NUM_EXPERTS,
    TARGET_M,
    TOP_K,
    add_vllm_root,
    gpu_median_us,
    load_extension,
    require_a100,
    weighted,
    write_csv,
)


def main(args: argparse.Namespace) -> None:
    add_vllm_root(args.vllm_root)
    import torch
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    require_a100()
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    load_extension(Path(__file__).resolve().parents[1])
    rows = []

    for m in TARGET_M:
        hidden = torch.empty((m, 1), device="cuda", dtype=torch.bfloat16)
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16)

        def baseline():
            return fused_topk(hidden, logits, TOP_K, renormalize=False)

        def optimized():
            weights = torch.empty((m, TOP_K), device="cuda", dtype=torch.float32)
            ids = torch.empty((m, TOP_K), device="cuda", dtype=torch.int32)
            source = torch.empty((m, TOP_K), device="cuda", dtype=torch.int32)
            torch.ops._moe_C.topk_softmax_a100_small(
                logits, weights, ids, source, False
            )
            return weights, ids, source

        reference = baseline()
        candidate = optimized()
        torch.cuda.synchronize()
        if not torch.equal(reference[1], candidate[1]):
            raise AssertionError(f"topk_ids mismatch at M={m}")
        if not torch.equal(reference[0], candidate[0]):
            raise AssertionError(f"topk_weights mismatch at M={m}")

        baseline_us = gpu_median_us(
            baseline, args.warmup, args.samples, args.rounds
        )
        optimized_us = gpu_median_us(
            optimized, args.warmup, args.samples, args.rounds
        )
        row = {
            "M": m,
            "frequency": DECODE_FREQUENCIES[m],
            "baseline_topk_us": baseline_us,
            "optimized_topk_us": optimized_us,
            "reduction_pct": (1.0 - optimized_us / baseline_us) * 100.0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    baseline_weighted = weighted(rows, "baseline_topk_us")
    optimized_weighted = weighted(rows, "optimized_topk_us")
    rows.append(
        {
            "M": "weighted",
            "frequency": sum(DECODE_FREQUENCIES.values()),
            "baseline_topk_us": baseline_weighted,
            "optimized_topk_us": optimized_weighted,
            "reduction_pct":
                (1.0 - optimized_weighted / baseline_weighted) * 100.0,
        }
    )
    output = args.output_dir / "topk_benchmark.csv"
    write_csv(output, rows)
    print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    main(parser.parse_args())
