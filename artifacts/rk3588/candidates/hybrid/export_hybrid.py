from pathlib import Path
from rknn.api import RKNN

root = Path(__file__).resolve().parents[4]
source = root / "artifacts/rk3588/yoloe-26s-rknn-source.onnx"
dataset = root / "artifacts/rk3588/candidates/int8/calibration.txt"
out = root / "candidates/hybrid/yoloe-26s-rk3588-hybrid.rknn"
out.parent.mkdir(parents=True, exist_ok=True)

rknn = RKNN(verbose=False)
try:
    assert rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
        quantized_dtype="w8a8",
        quantized_method="channel",
        float_dtype="float16",
    ) == 0
    assert rknn.load_onnx(model=str(source)) == 0
    assert rknn.build(do_quantization=True, dataset=str(dataset), auto_hybrid=True) == 0
    assert rknn.export_rknn(str(out)) == 0
finally:
    rknn.release()
print(out)
