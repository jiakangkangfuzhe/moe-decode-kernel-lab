#!/usr/bin/env python3
"""Measure the Top-k specialization inside the full routed MoE pipeline."""

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
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    require_a100()
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    load_extension(Path(__file__).resolve().parents[1])

    hidden_size, intermediate_size = 2048, 1408
    gate = torch.randn(
        (NUM_EXPERTS, hidden_size), device="cuda", dtype=torch.bfloat16
    ) / 20
    w1 = torch.randn(
        (NUM_EXPERTS, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    ) / 20
    w2 = torch.randn(
        (NUM_EXPERTS, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    ) / 20
    rows = []

    for m in TARGET_M:
        hidden = torch.randn(
            (m, hidden_size), device="cuda", dtype=torch.bfloat16
        ) / 10
        logits = torch.mm(hidden, gate.T)

        def pipeline(optimized: bool):
            if optimized:
                weights = torch.empty(
                    (m, TOP_K), device="cuda", dtype=torch.float32
                )
                ids = torch.empty((m, TOP_K), device="cuda", dtype=torch.int32)
                source = torch.empty(
                    (m, TOP_K), device="cuda", dtype=torch.int32
                )
                torch.ops._moe_C.topk_softmax_a100_small(
                    logits, weights, ids, source, False
                )
            else:
                weights, ids, _ = fused_topk(
                    hidden, logits, TOP_K, renormalize=False
                )
            return fused_experts(hidden, w1, w2, weights, ids)

        reference = pipeline(False)
        candidate = pipeline(True)
        torch.cuda.synchronize()
        max_error = float((reference.float() - candidate.float()).abs().max())
        if not torch.equal(reference, candidate):
            raise AssertionError(f"full MoE output mismatch at M={m}")

        baseline_us = gpu_median_us(
            lambda: pipeline(False), args.warmup, args.samples, args.rounds
        )
        optimized_us = gpu_median_us(
            lambda: pipeline(True), args.warmup, args.samples, args.rounds
        )
        row = {
            "M": m,
            "frequency": DECODE_FREQUENCIES[m],
            "baseline_full_moe_us": baseline_us,
            "optimized_full_moe_us": optimized_us,
            "gain_pct": (1.0 - optimized_us / baseline_us) * 100.0,
            "max_abs_error": max_error,
            "correctness": "PASS",
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    baseline_weighted = weighted(rows, "baseline_full_moe_us")
    optimized_weighted = weighted(rows, "optimized_full_moe_us")
    rows.append(
        {
            "M": "weighted",
            "frequency": sum(DECODE_FREQUENCIES.values()),
            "baseline_full_moe_us": baseline_weighted,
            "optimized_full_moe_us": optimized_weighted,
            "gain_pct":
                (1.0 - optimized_weighted / baseline_weighted) * 100.0,
            "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
            "correctness": "PASS",
        }
    )
    output = args.output_dir / "full_moe_benchmark.csv"
    write_csv(output, rows)
    print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    main(parser.parse_args())
