#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m gauge_detector benchmark --images data/benchmark/images --labels data/benchmark/labels.json --config configs/default.yaml --output artifacts/benchmark.csv
