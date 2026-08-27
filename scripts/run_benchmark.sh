#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/vllm [output-dir]" >&2
  exit 2
fi

vllm_root=$1
output_dir=${2:-results}

python scripts/benchmark_topk.py \
  --vllm-root "$vllm_root" --output-dir "$output_dir"
python scripts/benchmark_full_moe.py \
  --vllm-root "$vllm_root" --output-dir "$output_dir"
