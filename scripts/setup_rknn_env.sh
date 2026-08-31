#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PROJECT_DIR}/.venv-rknn"

python3.10 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/pip" install \
  "setuptools<82" \
  "numpy<2" \
  "PyYAML>=6" \
  "onnx>=1.16.1,<1.19.0" \
  "onnxruntime>=1.10.0" \
  "protobuf>=4.21.6,<=4.25.4" \
  "psutil>=5.9.0" \
  "ruamel.yaml>=0.17.21" \
  "scipy>=1.9.3" \
  "tqdm>=4.64.1" \
  "opencv-python>=4.5.5.64" \
  "fast-histogram>=0.11"
"${ENV_DIR}/bin/pip" install "rknn-toolkit2==2.3.2" --no-deps
"${ENV_DIR}/bin/python" -c \
  'import numpy, onnx, rknn; print("numpy", numpy.__version__); print("onnx", onnx.__version__); print("rknn", getattr(rknn, "__version__", "2.3.2"))'
