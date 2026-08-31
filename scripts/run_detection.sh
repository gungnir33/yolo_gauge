#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 可直接修改以下默认值；命令行参数会覆盖它们。
DEFAULT_INPUT="/home/mcl/data/yolo/检测采图"
DEFAULT_OUTPUT="$PROJECT_DIR/outputs/one_click"
DEFAULT_CONFIG="$PROJECT_DIR/configs/default.yaml"

INPUT_PATH="$DEFAULT_INPUT"
OUTPUT_PATH="$DEFAULT_OUTPUT"
CONFIG_PATH="$DEFAULT_CONFIG"
RECURSIVE=false

usage() {
    cat <<'EOF'
用法：
  bash scripts/run_detection.sh [选项]

选项：
  --input PATH    输入图片或图片文件夹（覆盖脚本顶部 DEFAULT_INPUT）
  --output PATH   输出文件夹（覆盖脚本顶部 DEFAULT_OUTPUT）
  --config PATH   配置文件（默认 configs/default.yaml）
  --recursive     输入为文件夹时递归搜索图片
  -h, --help      显示帮助

示例：
  bash scripts/run_detection.sh
  bash scripts/run_detection.sh --input "/path/to/image.jpg" --output "outputs/single"
  bash scripts/run_detection.sh --input "/path/to/images" --output "outputs/batch" --recursive
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "错误：参数 $1 缺少路径。" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            require_value "$@"
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            require_value "$@"
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --config)
            require_value "$@"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --recursive)
            RECURSIVE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误：未找到虚拟环境 Python：$PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -e "$INPUT_PATH" ]]; then
    echo "错误：输入路径不存在：$INPUT_PATH" >&2
    exit 2
fi
if [[ ! -f "$CONFIG_PATH" && -f "$PROJECT_DIR/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$PROJECT_DIR/$CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "错误：配置文件不存在：$CONFIG_PATH" >&2
    exit 2
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$INPUT_PATH" ]]; then
    if [[ "$RECURSIVE" == true ]]; then
        echo "提示：输入为单个文件，已忽略 --recursive。" >&2
    fi
    "$PYTHON_BIN" -m gauge_detector detect \
        --image "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --config "$CONFIG_PATH"
elif [[ -d "$INPUT_PATH" ]]; then
    command=(
        "$PYTHON_BIN" -m gauge_detector detect-dir
        --input "$INPUT_PATH"
        --output "$OUTPUT_PATH"
        --config "$CONFIG_PATH"
    )
    if [[ "$RECURSIVE" == true ]]; then
        command+=(--recursive)
    fi
    "${command[@]}"
else
    echo "错误：输入路径既不是普通文件也不是文件夹：$INPUT_PATH" >&2
    exit 2
fi
