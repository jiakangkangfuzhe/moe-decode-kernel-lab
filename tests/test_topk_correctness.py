"""Clean-room functional tests for the final CUDA TopK4 implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


TARGET_M = (1, 2, 4, 8, 16, 32, 64)
EXPERTS = 60
TOP_K = 4


@pytest.fixture(scope="session", autouse=True)
def extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the published kernel is restricted to SM80")
    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = (
        f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    root = Path(__file__).resolve().parents[1]
    return load(
        name="moe_decode_topk_test_ext",
        sources=[
            str(root / "src/topk/bindings.cpp"),
            str(root / "src/topk/topk_final_sprint.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def run_candidate(logits: torch.Tensor):
    rows = logits.size(0)
    weights = torch.empty((rows, TOP_K), device="cuda", dtype=torch.float32)
    ids = torch.empty((rows, TOP_K), device="cuda", dtype=torch.int32)
    source = torch.empty((rows, TOP_K), device="cuda", dtype=torch.int32)
    torch.ops._moe_C.topk_softmax_a100_small(
        logits, weights, ids, source, False
    )
    torch.cuda.synchronize()
    return weights, ids, source


@pytest.mark.parametrize("rows", TARGET_M)
def test_random_logits_match_softmax_topk(extension, rows: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(7000 + rows)
    logits = torch.randn(
        (rows, EXPERTS), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    weights, ids, source = run_candidate(logits)
    probabilities = torch.softmax(logits.float(), dim=-1)
    expected_ids = torch.argsort(
        probabilities, dim=-1, descending=True, stable=True
    )[:, :TOP_K].to(torch.int32)
    expected_weights = torch.gather(
        probabilities, 1, expected_ids.to(torch.int64)
    )
    expected_source = (
        torch.arange(TOP_K, device="cuda", dtype=torch.int32)[:, None] * rows
        + torch.arange(rows, device="cuda", dtype=torch.int32)[None, :]
    ).T.contiguous()
    assert torch.equal(ids, expected_ids)
    assert torch.equal(source, expected_source)
    assert torch.allclose(weights, expected_weights, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("rows", TARGET_M)
def test_ties_choose_lower_expert_ids(extension, rows: int) -> None:
    logits = torch.zeros((rows, EXPERTS), device="cuda", dtype=torch.bfloat16)
    weights, ids, _ = run_candidate(logits)
    expected_ids = torch.arange(TOP_K, device="cuda", dtype=torch.int32).repeat(
        rows, 1
    )
    assert torch.equal(ids, expected_ids)
    assert torch.allclose(
        weights,
        torch.full_like(weights, 1.0 / EXPERTS),
        atol=2e-6,
        rtol=2e-6,
    )


def test_unsupported_expert_count_is_rejected(extension) -> None:
    logits = torch.zeros((1, 65), device="cuda", dtype=torch.bfloat16)
    weights = torch.empty((1, TOP_K), device="cuda", dtype=torch.float32)
    ids = torch.empty((1, TOP_K), device="cuda", dtype=torch.int32)
    source = torch.empty_like(ids)
    with pytest.raises(RuntimeError, match="4<=E<=64"):
        torch.ops._moe_C.topk_softmax_a100_small(
            logits, weights, ids, source, False
        )
