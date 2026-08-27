"""Shared helpers for building and timing the standalone CUDA extension."""

from __future__ import annotations

import csv
import os
import statistics
import sys
from pathlib import Path
from typing import Callable


TARGET_M = (1, 2, 4, 8, 16, 32, 64)
NUM_EXPERTS = 60
TOP_K = 4
DECODE_FREQUENCIES = {
    1: 3072,
    2: 3024,
    4: 3024,
    8: 3024,
    16: 3048,
    32: 3000,
    64: 2976,
}


def add_vllm_root(vllm_root: Path) -> None:
    """Make a checked-out vLLM source tree importable without installation."""
    root = vllm_root.expanduser().resolve()
    if not (root / "vllm").is_dir():
        raise ValueError(f"not a vLLM source tree: {root}")
    sys.path.insert(0, str(root))


def load_extension(repo_root: Path, build_directory: Path | None = None):
    """JIT-build the final standalone op for SM80."""
    from torch.utils.cpp_extension import load

    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = (
        f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    source_root = repo_root / "src/topk"
    if build_directory is not None:
        build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="moe_decode_topk_ext",
        sources=[
            str(source_root / "bindings.cpp"),
            str(source_root / "topk_final_sprint.cu"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=(
            str(build_directory.resolve()) if build_directory is not None else None
        ),
        verbose=False,
    )


def require_a100() -> None:
    """Reject benchmarks outside the validated SM80 target."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("this benchmark is restricted to CUDA SM80")


def gpu_median_us(
    fn: Callable[[], object], warmup: int, samples: int, rounds: int
) -> float:
    """Return the median of independent round medians using CUDA events."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    round_medians = []
    for _ in range(rounds):
        values = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) * 1000.0)
        round_medians.append(statistics.median(values))
    return statistics.median(round_medians)


def weighted(rows: list[dict], field: str) -> float:
    """Weight a latency field by the captured Decode M distribution."""
    total = sum(int(row["frequency"]) for row in rows)
    return sum(int(row["frequency"]) * float(row[field]) for row in rows) / total


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a small result table with stable column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
