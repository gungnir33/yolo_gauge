#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_DIR}/configs/rk3588.yaml"
MODEL=""
INPUT=""
OUTPUT="${PROJECT_DIR}/outputs/rknn"
CORE_MASK="AUTO"
PYTHON_BIN="${GAUGE_PYTHON:-python3}"
RECURSIVE=0

usage() {
  cat <<'EOF'
RK3588 Python 仪表检测

用法:
  scripts/run_rknn_detection.sh --input PATH [选项]

选项:
  --model PATH       RKNN 模型位置；默认使用配置文件中的 rknn_path
  --input PATH       输入图片或图片目录
  --output PATH      输出目录，默认 outputs/rknn
  --config PATH      RKNN YAML 配置，默认 configs/rk3588.yaml
  --core-mask NAME   AUTO、CORE_0、CORE_1、CORE_2、CORE_0_1 或 CORE_0_1_2
  --python PATH      板端 Python 解释器，默认 python3；也可设置 GAUGE_PYTHON
  --recursive        递归检测输入目录
  -h, --help         显示帮助
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL="${2:?--model 需要路径}"; shift 2 ;;
    --input) INPUT="${2:?--input 需要路径}"; shift 2 ;;
    --output) OUTPUT="${2:?--output 需要路径}"; shift 2 ;;
    --config) CONFIG="${2:?--config 需要路径}"; shift 2 ;;
    --core-mask) CORE_MASK="${2:?--core-mask 需要名称}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python 需要路径}"; shift 2 ;;
    --recursive) RECURSIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${INPUT}" ]]; then
  echo "必须提供 --input" >&2
  exit 2
fi
if [[ ! -e "${INPUT}" ]]; then
  echo "输入路径不存在: ${INPUT}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "配置文件不存在: ${CONFIG}" >&2
  exit 2
fi
if [[ -n "${MODEL}" && ! -f "${MODEL}" ]]; then
  echo "RKNN 模型不存在: ${MODEL}" >&2
  exit 2
fi

INPUT="$(cd "$(dirname "${INPUT}")" && pwd -P)/$(basename "${INPUT}")"
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd -P)/$(basename "${CONFIG}")"
if [[ -n "${MODEL}" ]]; then
  MODEL="$(cd "$(dirname "${MODEL}")" && pwd -P)/$(basename "${MODEL}")"
fi

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/gauge-rknn.XXXXXX.yaml")"
trap 'rm -f -- "${TEMP_CONFIG}"' EXIT
"${PYTHON_BIN}" -c '
import sys, yaml
source, destination, model, core_mask = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
model_config = config.setdefault("model", {})
if model:
    model_config["rknn_path"] = model
model_config["core_mask"] = core_mask
with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
' "${CONFIG}" "${TEMP_CONFIG}" "${MODEL}" "${CORE_MASK}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -d "${INPUT}" ]]; then
  ARGS=(detect-dir --input "${INPUT}" --output "${OUTPUT}" --config "${TEMP_CONFIG}")
  if ((RECURSIVE)); then ARGS+=(--recursive); fi
else
  ARGS=(detect --image "${INPUT}" --output "${OUTPUT}" --config "${TEMP_CONFIG}")
fi
"${PYTHON_BIN}" -m gauge_detector "${ARGS[@]}"
