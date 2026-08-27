#!/usr/bin/env python3
"""Aggregate three-run Serving A/B CSVs using median metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


METRICS = ("tpot_median_ms", "itl_p99_ms", "output_tokens_per_s")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(args: argparse.Namespace) -> None:
    inputs = {
        "baseline": read(args.baseline),
        "optimized": read(args.optimized),
    }
    rows = []
    for concurrency in (8, 16, 32):
        medians = {}
        for implementation, source in inputs.items():
            selected = [
                row for row in source if int(row["concurrency"]) == concurrency
            ]
            if len(selected) != 3:
                raise ValueError(
                    f"expected three {implementation} runs at C{concurrency}"
                )
            medians[implementation] = {
                metric: statistics.median(float(row[metric]) for row in selected)
                for metric in METRICS
            }
        baseline, optimized = medians["baseline"], medians["optimized"]
        rows.append(
            {
                "concurrency": concurrency,
                "baseline_tpot_median_ms": baseline["tpot_median_ms"],
                "optimized_tpot_median_ms": optimized["tpot_median_ms"],
                "tpot_gain_pct":
                    (1 - optimized["tpot_median_ms"] / baseline["tpot_median_ms"])
                    * 100,
                "p99_itl_gain_pct":
                    (1 - optimized["itl_p99_ms"] / baseline["itl_p99_ms"])
                    * 100,
                "output_tokens_per_s_gain_pct":
                    (optimized["output_tokens_per_s"]
                     / baseline["output_tokens_per_s"] - 1)
                    * 100,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/serving_summary.csv"))
    main(parser.parse_args())
