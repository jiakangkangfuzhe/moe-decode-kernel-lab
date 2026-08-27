#!/usr/bin/env python3
"""Build the standalone CUDA op and print its shared-library path."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_extension, require_a100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build/topk"))
    args = parser.parse_args()
    require_a100()
    module = load_extension(Path(__file__).resolve().parents[1], args.build_dir)
    print(module.__file__)
