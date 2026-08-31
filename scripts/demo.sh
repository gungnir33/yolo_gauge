#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m gauge_detector detect --image /home/mcl/data/yolo/检测采图/detect-02.jpg --output outputs --config configs/default.yaml
