#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q tests/test_topk_correctness.py
