from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from .benchmark import evaluate, recommend, save_benchmark_csv
from .crop import save_crops
from .detector import GaugeDetector
from .export import export_model
from .io_utils import read_image, result_to_dict, save_json
from .prompt_profile import prepare_prompt_profile
from .visualization import draw_detections

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration path")
    parser.add_argument("--verbose", action="store_true")


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
            raise OSError(f"Failed to save annotated image: {annotated_path}")
    if cfg["output"]["save_json"]:
        save_json(result_to_dict(result), output_dir / f"{stem}.json")
    if cfg["output"]["save_crops"]:
        save_crops(image, result.detections, output_dir / "crops", stem)
    return result, annotated_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gauge_detector", description="YOLOE text-prompt instrument detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Detect instruments in one image")
    detect.add_argument("--image", required=True)
    detect.add_argument("--output", default="outputs")
    _add_common(detect)

    detect_dir = subparsers.add_parser("detect-dir", help="Detect instruments in all images in a directory")
    detect_dir.add_argument("--input", required=True)
    detect_dir.add_argument("--output", default="outputs")
    detect_dir.add_argument("--recursive", action="store_true")
    _add_common(detect_dir)

    benchmark = subparsers.add_parser("benchmark", help="Evaluate text-prompt detection on a labeled image set")
    benchmark.add_argument("--images", required=True)
    benchmark.add_argument("--labels", required=True)
    benchmark.add_argument("--output", default="artifacts/benchmark.csv")
    benchmark.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    _add_common(benchmark)

    export = subparsers.add_parser("export", help="Export a text-prompt ONNX or TensorRT model")
    export.add_argument("--format", choices=["engine", "onnx"], required=True)
    _add_common(export)

    prepare_profile = subparsers.add_parser("prepare-profile", help="Save static YOLOE text prompt embeddings")
    prepare_profile.add_argument("--output", required=True, help="Output .npz prompt profile path")
    _add_common(prepare_profile)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "detect":
            detector = GaugeDetector(args.config)
            result, annotated = _save_detection(detector, Path(args.image), Path(args.output))
            print(
                f"Image: {args.image}\nDetected instruments: {len(result.detections)}\n"
                f"Inference: {result.inference_ms:.2f} ms\nResults: {annotated}"
            )
        elif args.command == "detect-dir":
            detector = GaugeDetector(args.config)
            directory = Path(args.input)
            iterator = directory.rglob("*") if args.recursive else directory.glob("*")
            paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            for path in paths:
                result, _ = _save_detection(detector, path, Path(args.output))
                print(f"{path}: {len(result.detections)} instrument(s)")
            print(f"Processed images: {len(paths)}")
        elif args.command == "benchmark":
            import yaml

            search_path = Path(args.benchmark_config)
            with search_path.open("r", encoding="utf-8") as handle:
                search = yaml.safe_load(handle) or {}
            detector = GaugeDetector(args.config)
            runtime_model = detector.config["model"]["name"]
            configured_models = search.get("models", [runtime_model])
            if runtime_model not in configured_models:
                raise ValueError(f"Benchmark config does not include runtime model: {runtime_model}")
            rows = []
            for imgsz in search.get("image_sizes", [detector.model.imgsz]):
                detector.model.imgsz = int(imgsz)
                detector.config["model"]["imgsz"] = int(imgsz)
                detector.model.warmup(int(search.get("warmup_runs", 3)))
                for conf in search.get("confidences", [detector.config["detection"]["conf"]]):
                    detector.config["detection"]["conf"] = float(conf)
                    metrics = evaluate(detector, args.images, args.labels, float(search.get("iou_threshold", 0.5)))
                    rows.append({"model": runtime_model, "imgsz": imgsz, "conf": conf, **metrics})
            output = save_benchmark_csv(rows, args.output)
            best = recommend(rows, float(search.get("target_recall", 0.95)))
            skipped = [name for name in configured_models if name != runtime_model]
            if skipped:
                logging.getLogger(__name__).warning(
                    "Skipped models not matching the configured runtime model: %s.", skipped
                )
            print(f"Recommended: {best}\nCSV: {output}")
        elif args.command == "export":
            output = export_model(args.config, args.format)
            print(f"Exported: {output}")
        elif args.command == "prepare-profile":
            profile, metadata = prepare_prompt_profile(args.config, args.output)
            print(f"Prompt profile: {profile}\nMetadata: {metadata}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
