#!/usr/bin/env python3
"""Summarize launch M separately from routed per-expert GEMM M."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLDS = (1, 2, 4, 8, 16, 32, 64)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else (
        values[lo] * (hi - position) + values[hi] * (position - lo)
    )


def phase_summary(phase: str, workload: list[dict], experts: list[dict]) -> list[str]:
    calls = [row for row in workload if row["phase"] == phase]
    expert_rows = [row for row in experts if row["phase"] == phase]
    launch_m = [int(row["M"]) for row in calls]
    active_m = [
        int(row["token_count"])
        for row in expert_rows
        if int(row["token_count"]) > 0
    ]
    if not calls or not active_m:
        return [f"## {phase.title()}", "", "No captured calls.", ""]

    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row in expert_rows:
        key = (row["timestamp"], row["rank"], row["call_index"], row["layer"])
        grouped[key].append(int(row["token_count"]))
    imbalance = []
    for counts in grouped.values():
        active = [count for count in counts if count > 0]
        if active:
            imbalance.append(max(active) / statistics.mean(active))
    configs = Counter((row["config_source"], row["kernel_config"]) for row in calls)
    padding = [float(row["padding_ratio"]) for row in calls]
    active_ratio = [float(row["active_expert_ratio"]) for row in calls]

    lines = [
        f"## {phase.title()}", "", "| metric | value |", "|---|---:|",
        f"| layer calls | {len(calls)} |",
        f"| launch M median / P90 | {percentile(launch_m, .5):.2f} / {percentile(launch_m, .9):.2f} |",
        f"| active-expert M median / P90 / max | {percentile(active_m, .5):.2f} / {percentile(active_m, .9):.2f} / {max(active_m)} |",
        f"| active expert ratio median / P90 | {percentile(active_ratio, .5) * 100:.2f}% / {percentile(active_ratio, .9) * 100:.2f}% |",
        f"| max/mean active load median / P90 | {percentile(imbalance, .5):.2f}x / {percentile(imbalance, .9):.2f}x |",
        f"| padding overhead median / P90 | {percentile(padding, .5) * 100:.2f}% / {percentile(padding, .9) * 100:.2f}% |",
        "",
        "Active-expert-M CDF: " + ", ".join(
            f"<= {threshold}: {sum(value <= threshold for value in active_m) / len(active_m) * 100:.2f}%"
            for threshold in THRESHOLDS
        ),
        "", "Kernel configurations:", "",
    ]
    lines.extend(
        f"- {count} calls: `{source}` / `{config}`"
        for (source, config), count in configs.most_common()
    )
    lines.append("")
    return lines


def main(args: argparse.Namespace) -> None:
    with args.input.open(newline="", encoding="utf-8") as handle:
        workload = list(csv.DictReader(handle))
    with args.expert_input.open(newline="", encoding="utf-8") as handle:
        experts = list(csv.DictReader(handle))
    lines = [
        "# MoE workload analysis", "",
        "`M` is the total-token launch M; active-expert M is the routed row count per expert.", "",
    ]
    lines += phase_summary("prefill", workload, experts)
    lines += phase_summary("decode", workload, experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expert-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/workload_analysis.md"))
    main(parser.parse_args())
