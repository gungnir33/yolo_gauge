from pathlib import Path
import onnx
from onnx import helper, numpy_helper
import numpy as np
from rknn.api import RKNN

base = Path(__file__).resolve().parents[4]
src = base / "artifacts/rk3588/yoloe-26s-rknn-source.onnx"
calib = base / "artifacts/rk3588/candidates/int8/calibration.txt"
outdir = base / "artifacts/rk3588/candidates/split"
outdir.mkdir(parents=True, exist_ok=True)
split = outdir / "yoloe-26s-rk3588-split.onnx"
rknn_out = outdir / "yoloe-26s-rk3588-split-int8.rknn"

model = onnx.load(str(src))
old = model.graph.output[0]
old_name = old.name
for init in [numpy_helper.from_array(np.array([0], np.int64), "split_starts_box"),
             numpy_helper.from_array(np.array([4], np.int64), "split_ends_box"),
             numpy_helper.from_array(np.array([4], np.int64), "split_starts_score"),
             numpy_helper.from_array(np.array([9], np.int64), "split_ends_score"),
             numpy_helper.from_array(np.array([1], np.int64), "split_axes"),
             numpy_helper.from_array(np.array([1], np.int64), "split_steps")]:
    model.graph.initializer.append(init)
model.graph.node.extend([
    helper.make_node("Slice", [old_name, "split_starts_box", "split_ends_box", "split_axes", "split_steps"], ["boxes_out"]),
    helper.make_node("Slice", [old_name, "split_starts_score", "split_ends_score", "split_axes", "split_steps"], ["scores_out"]),
])
model.graph.output.remove(old)
model.graph.output.extend([
    helper.make_tensor_value_info("boxes_out", onnx.TensorProto.FLOAT, [1, 4, 10710]),
    helper.make_tensor_value_info("scores_out", onnx.TensorProto.FLOAT, [1, 5, 10710]),
])
onnx.checker.check_model(model)
onnx.save(model, str(split))

r = RKNN(verbose=False)
try:
    assert r.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform="rk3588") == 0
    assert r.load_onnx(model=str(split)) == 0
    assert r.build(do_quantization=True, dataset=str(calib)) == 0
    assert r.export_rknn(str(rknn_out)) == 0
finally:
    r.release()
print(split)
print(rknn_out)
