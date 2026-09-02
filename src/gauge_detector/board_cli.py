from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from .crop import save_crops
from .detector import GaugeDetector
from .io_utils import read_image, result_to_dict, save_json
from .visualization import draw_detections

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/rk3588.yaml", help="RK3588 YAML 配置路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")


def _save_detection(detector: GaugeDetector, image_path: Path, output_dir: Path):
    image = read_image(image_path)
    result = detector.detect_array(image)
    result.image_path = str(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cfg = detector.config
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    if cfg["output"]["save_annotated_image"]:
        annotated = draw_detections(
            image,
            result.detections,
            thickness=int(cfg["visualization"]["thickness"]),
            draw_numbers=True,
        )
        if not cv2.imwrite(str(annotated_path), annotated):
            raise OSError(f"保存标注图片失败: {annotated_path}")
    if cfg["output"]["save_json"]:
        save_json(result_to_dict(result), output_dir / f"{stem}.json")
    if cfg["output"]["save_crops"]:
        save_crops(image, result.detections, output_dir / "crops", stem)
    return result, annotated_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gauge_detector.board_cli",
        description="RK3588 RKNN 工业仪表检测",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="检测单张图片")
    detect.add_argument("--image", required=True)
    detect.add_argument("--output", default="outputs/rknn")
    _add_common(detect)

    detect_dir = subparsers.add_parser("detect-dir", help="检测目录中的全部图片")
    detect_dir.add_argument("--input", required=True)
    detect_dir.add_argument("--output", default="outputs/rknn")
    detect_dir.add_argument("--recursive", action="store_true")
    _add_common(detect_dir)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        with GaugeDetector(args.config) as detector:
            if args.command == "detect":
                result, annotated = _save_detection(detector, Path(args.image), Path(args.output))
                print(
                    f"图片: {args.image}\n检测到仪表: {len(result.detections)}\n"
                    f"RKNN 推理: {result.inference_ms:.2f} ms\n结果: {annotated}"
                )
                return

            directory = Path(args.input)
            iterator = directory.rglob("*") if args.recursive else directory.glob("*")
            paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            for path in paths:
                result, _ = _save_detection(detector, path, Path(args.output))
                print(f"{path}: {len(result.detections)} 个仪表，RKNN {result.inference_ms:.2f} ms")
            print(f"处理图片: {len(paths)} 张")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"错误: {exc}\n")


if __name__ == "__main__":
    main()
