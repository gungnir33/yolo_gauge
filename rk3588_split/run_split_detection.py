#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np

_root = Path(__file__).resolve().parents[1]
_sources = [Path(os.environ["GAUGE_SOURCE"])] if os.environ.get("GAUGE_SOURCE") else []
_sources += [_root / "yoloe_gauge_detector" / "src", _root / "yolo_gauge_rk3588_python" / "src"]
SOURCE = next((p for p in _sources if (p / "gauge_detector").is_dir()), _sources[0])
sys.path.insert(0, str(SOURCE))
from gauge_detector.preprocess import letterbox_rgb, onnx_tensor
from gauge_detector.runtime_output import decode_raw_output
from gauge_detector.postprocess import remove_duplicate_boxes, select_single_target, sort_detections

COLORS = [(255, 0, 0), (0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def process(path, session, backend, output_dir, conf, iou):
    image = cv2.imread(str(path))
    if image is None: raise ValueError(f"cannot read image: {path}")
    if backend == "onnx":
        tensor, transform = onnx_tensor(image, (544, 960), (114, 114, 114))
        started = time.perf_counter(); raw = session.run(None, {session.get_inputs()[0].name: tensor}); elapsed = (time.perf_counter()-started)*1000
    else:
        rgb, transform = letterbox_rgb(image, (544, 960), (114, 114, 114)); inp = np.expand_dims(rgb, 0)
        started = time.perf_counter(); raw = session.inference(inputs=[inp], data_format=["nhwc"]); elapsed = (time.perf_counter()-started)*1000
    values = np.concatenate([np.asarray(raw[0]), np.asarray(raw[1])], axis=1)
    detections = decode_raw_output(values, transform, image.shape, conf, iou, 20)
    detections = remove_duplicate_boxes(detections, iou)
    detections = select_single_target(detections, {"enabled": True, "containment_threshold": 0.90})
    detections = sort_detections(detections)
    canvas = image.copy(); rows=[]
    for idx, d in enumerate(detections):
        color = COLORS[idx % len(COLORS)]; x1,y1,x2,y2 = map(int, (d.x1,d.y1,d.x2,d.y2)); cv2.rectangle(canvas,(x1,y1),(x2,y2),color,3); cv2.putText(canvas,str(idx+1),(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2,cv2.LINE_AA)
        rows.append({"id":idx,"class_name":"instrument","confidence":round(float(d.confidence),4),"bbox_xyxy":[x1,y1,x2,y2],"center":[round(float(d.center[0]),2),round(float(d.center[1]),2)]})
    output_dir.mkdir(parents=True, exist_ok=True); stem=path.stem
    cv2.imwrite(str(output_dir/f"{stem}_annotated.jpg"), canvas)
    payload={"image":str(path),"width":int(image.shape[1]),"height":int(image.shape[0]),"inference_ms":round(elapsed,2),"num_detections":len(rows),"detections":rows}
    (output_dir/f"{stem}.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return payload

def main():
    p=argparse.ArgumentParser(); p.add_argument("--backend",choices=["rknn","onnx"],required=True); p.add_argument("--model",required=True); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--conf",type=float,default=0.15); p.add_argument("--iou",type=float,default=0.5); a=p.parse_args()
    if a.backend == "rknn":
        from rknnlite.api import RKNNLite
        session=RKNNLite(); assert session.load_rknn(a.model)==0; assert session.init_runtime()==0
    else:
        import onnxruntime as ort; session=ort.InferenceSession(a.model,providers=["CPUExecutionProvider"])
    src=Path(a.input); paths=sorted([src] if src.is_file() else [x for x in src.iterdir() if x.suffix.lower() in EXTS])
    out=Path(a.output); results=[]
    try:
        for path in paths:
            try:
                r=process(path,session,a.backend,out,a.conf,a.iou); results.append(r); print(f"{path.name}\tdetections={r['num_detections']}\tinference_ms={r['inference_ms']:.2f}")
            except Exception as exc: print(f"{path.name}\tERROR: {exc}",file=sys.stderr)
    finally:
        if a.backend == "rknn": session.release()
    if results:
        avg=sum(r["inference_ms"] for r in results)/len(results); (out/"summary.json").write_text(json.dumps({"backend":a.backend,"count":len(results),"avg_inference_ms":round(avg,2),"results":results},ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(f"average_inference_ms={avg:.2f}")
if __name__ == "__main__": main()
