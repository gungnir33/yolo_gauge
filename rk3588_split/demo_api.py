#!/usr/bin/env python3
"""GaugeDetectorAPI 调用示例。"""
import argparse
import json
from pathlib import Path

from detector_api import GaugeDetectorAPI


def main():
    parser = argparse.ArgumentParser(description="调用 RKNN 仪表检测 Python 接口")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--model", default=None, help="可选：指定 RKNN 模型路径")
    parser.add_argument("--as-bytes", action="store_true", help="先读入 JPEG bytes，再以内存数据调用接口")
    args = parser.parse_args()

    with GaugeDetectorAPI(model_path=args.model) as detector:
        image_input = Path(args.image).read_bytes() if args.as_bytes else args.image
        result_json = detector.detect(image_input)

    # 先解析再格式化，确保接口返回值确实是合法 JSON 字符串。
    print(json.dumps(json.loads(result_json), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
