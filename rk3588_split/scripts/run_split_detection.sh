#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND=""; MODEL=""; INPUT=""; OUTPUT=""
while [[ $# -gt 0 ]]; do case "$1" in --backend|--model|--input|--output) [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }; v="$2"; case "$1" in --backend) BACKEND="$v";; --model) MODEL="$v";; --input) INPUT="$v";; --output) OUTPUT="$v";; esac; shift 2;; -h|--help) echo "usage: $0 --backend rknn|onnx --model MODEL --input IMAGE_DIR --output OUT"; exit 0;; *) echo "unknown option $1" >&2; exit 2;; esac; done
[[ -n "$BACKEND" && -n "$MODEL" && -n "$INPUT" && -n "$OUTPUT" ]] || { echo "--backend/--model/--input/--output are required" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT/run_split_detection.py" --backend "$BACKEND" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
