"""面向外部 Python 程序的 RKNN 仪表检测接口。"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sources = []
if os.environ.get("GAUGE_SOURCE"):
    sources.append(Path(os.environ["GAUGE_SOURCE"]))
sources += [ROOT / "yoloe_gauge_detector" / "src", ROOT.parent / "yolo_gauge_rk3588_python" / "src"]
SOURCE = next((p for p in sources if (p / "gauge_detector").is_dir()), None)
if SOURCE is None:
    raise ImportError("无法找到 gauge_detector 源码，请设置 GAUGE_SOURCE。")
sys.path.insert(0, str(SOURCE))

from gauge_detector.postprocess import remove_duplicate_boxes, select_single_target, sort_detections
from gauge_detector.preprocess import letterbox_rgb
from gauge_detector.runtime_output import decode_raw_output


class GaugeDetectorAPI:
    """复用同一个 RKNN 会话，支持图片路径或 JPEG bytes 输入。"""

    def __init__(self, model_path=None, conf=0.15, iou=0.50):
        from rknnlite.api import RKNNLite

        self.model_path = Path(model_path) if model_path else ROOT / "models/yoloe-26s-rk3588-split-int8.rknn"
        self.conf = float(conf)
        self.iou = float(iou)
        self.runtime = RKNNLite()
        if self.runtime.load_rknn(str(self.model_path)) != 0 or self.runtime.init_runtime() != 0:
            self.runtime.release()
            raise RuntimeError("RKNN 模型加载或初始化失败")
        self.closed = False

    @staticmethod
    def _read_image(image_input):
        """读取路径或编码图片 bytes，返回 BGR 图像及输入标识。"""
        if isinstance(image_input, (bytes, bytearray, memoryview)):
            buffer = np.frombuffer(image_input, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码 JPEG bytes")
            return image, "<jpeg_bytes>"
        path = Path(image_input)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError("无法读取图片: %s" % path)
        return image, str(path)

    def detect(self, image_input):
        """检测单张图片并返回 JSON 字符串；输入可为路径或 JPEG bytes。"""
        if self.closed:
            raise RuntimeError("检测器已关闭")
        image, image_name = self._read_image(image_input)
        rgb, transform = letterbox_rgb(image, (544, 960), (114, 114, 114))
        started = time.perf_counter()
        raw = self.runtime.inference(inputs=[np.expand_dims(rgb, 0)], data_format=["nhwc"])
        elapsed = (time.perf_counter() - started) * 1000
        values = np.concatenate([np.asarray(raw[0]), np.asarray(raw[1])], axis=1)
        detections = decode_raw_output(values, transform, image.shape, self.conf, self.iou, 20)
        detections = sort_detections(select_single_target(remove_duplicate_boxes(detections, self.iou), {"enabled": True, "containment_threshold": 0.90}))
        rows = []
        for index, item in enumerate(detections):
            rows.append({
                "id": index,
                "class_id": 0,
                "class_name": "instrument",
                "confidence": round(float(item.confidence), 4),
                "bbox_xyxy": [round(float(v), 2) for v in item.xyxy],
                "center": [round(float(item.center[0]), 2), round(float(item.center[1]), 2)],
            })
        return json.dumps({"image": image_name, "width": int(image.shape[1]), "height": int(image.shape[0]), "inference_ms": round(elapsed, 2), "num_detections": len(rows), "detections": rows}, ensure_ascii=False)

    def close(self):
        if not self.closed:
            self.runtime.release()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def detect_image(image_input, model_path=None):
    """一次性检测接口；输入可为路径或 JPEG bytes。"""
    with GaugeDetectorAPI(model_path=model_path) as detector:
        return detector.detect(image_input)
