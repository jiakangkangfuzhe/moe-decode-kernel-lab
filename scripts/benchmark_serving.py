#!/usr/bin/env python3
"""Small final Serving client for an already running OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else (
        values[lo] * (hi - position) + values[hi] * (position - lo)
    )


async def request(client, url: str, model: str, prompt: list[int], output: int):
    start = time.perf_counter()
    first = None
    token_times = []
    token_count = 0
    async with client.stream(
        "POST",
        f"{url.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": output,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=600,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                token_count = int(event["usage"].get("completion_tokens", 0))
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if first is None:
                    first = now
                token_times.append(now)
    end = time.perf_counter()
    if not token_count:
        token_count = len(token_times)
    if token_count != output or first is None:
        raise RuntimeError(f"expected {output} streamed tokens, got {token_count}")
    itls = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    return {
        "ttft_ms": (first - start) * 1000,
        "tpot_ms": (end - first) * 1000 / max(output - 1, 1),
        "itls_ms": itls,
        "latency_ms": (end - start) * 1000,
    }


async def run(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed = tokenizer.encode(
        "Sparse mixture of experts decode performance on NVIDIA GPUs. ",
        add_special_tokens=False,
    )
    rows = []
    limits = httpx.Limits(max_connections=max(args.concurrency) + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        for run_index in range(args.runs):
            for concurrency in args.concurrency:
                prompts = []
                for request_index in range(concurrency):
                    suffix = tokenizer.encode(
                        f" run {run_index} request {request_index}",
                        add_special_tokens=False,
                    )
                    body = args.prompt_len - len(suffix)
                    prompts.append(
                        (seed * math.ceil(body / len(seed)))[:body] + suffix
                    )
                started = time.perf_counter()
                results = await asyncio.gather(
                    *[
                        request(
                            client, args.base_url, args.model, prompt,
                            args.output_len
                        )
                        for prompt in prompts
                    ]
                )
                elapsed = time.perf_counter() - started
                itls = [value for result in results for value in result["itls_ms"]]
                rows.append(
                    {
                        "implementation": args.implementation,
                        "concurrency": concurrency,
                        "run": run_index,
                        "tpot_median_ms": statistics.median(
                            result["tpot_ms"] for result in results
                        ),
                        "itl_p99_ms": percentile(itls, 0.99),
                        "output_tokens_per_s":
                            concurrency * args.output_len / elapsed,
                    }
                )
                print(rows[-1], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--implementation", choices=("baseline", "optimized"), required=True
    )
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
