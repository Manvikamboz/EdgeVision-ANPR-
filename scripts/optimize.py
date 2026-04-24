"""
EdgeSight – ONNX Export & INT8 Quantization
=============================================
Run after training to optimize the model for edge deployment.

Usage
-----
# Export to ONNX + quantize
python scripts/optimize.py --weights models/best_vehicle.pt

# Benchmark PyTorch vs ONNX vs INT8
python scripts/optimize.py --weights models/best_vehicle.pt --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeSight – Model Optimization")
    p.add_argument("--weights", required=True, help="Path to .pt file")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--benchmark", action="store_true", help="Compare runtimes")
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def export_onnx(weights: str, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO
    model = YOLO(weights)
    model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    onnx_path = Path(weights).with_suffix(".onnx")
    print(f"✓ ONNX exported → {onnx_path}  ({onnx_path.stat().st_size/1e6:.2f} MB)")
    return onnx_path


def quantize_int8(onnx_path: Path) -> Path:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out_path = onnx_path.parent / (onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    print(f"✓ INT8 quantised  → {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    return out_path


def benchmark(weights: str, onnx_path: Path, int8_path: Path, imgsz: int, iters: int) -> None:
    import onnxruntime as ort
    from ultralytics import YOLO

    dummy_np  = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    dummy_pt  = torch.from_numpy(dummy_np)

    def time_runs(fn, label: str, n: int = iters) -> float:
        for _ in range(10): fn()  # warm up
        t0 = time.perf_counter()
        for _ in range(n): fn()
        elapsed = (time.perf_counter() - t0) * 1000 / n
        print(f"  {label:<25} {elapsed:7.2f} ms  ({1000/elapsed:.1f} FPS)")
        return elapsed

    # PyTorch
    pt_model = YOLO(weights)
    def run_pt(): pt_model.predict(dummy_np, imgsz=imgsz, verbose=False)

    # ONNX FP32
    sess_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name   = sess_fp32.get_inputs()[0].name
    def run_fp32(): sess_fp32.run(None, {in_name: dummy_np})

    # ONNX INT8
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    in8_name  = sess_int8.get_inputs()[0].name
    def run_int8(): sess_int8.run(None, {in8_name: dummy_np})

    print(f"\n─── Benchmark ({iters} iterations, {imgsz}×{imgsz}) ─────────────")
    t_pt   = time_runs(run_pt,   "PyTorch FP32")
    t_fp32 = time_runs(run_fp32, "ONNX FP32")
    t_int8 = time_runs(run_int8, "ONNX INT8")
    print(f"\n  Speedup ONNX/PT  : {t_pt/t_fp32:.2f}×")
    print(f"  Speedup INT8/PT  : {t_pt/t_int8:.2f}×")
    print(f"  Speedup INT8/FP32: {t_fp32/t_int8:.2f}×")


if __name__ == "__main__":
    args = parse_args()

    onnx_path = export_onnx(args.weights, args.imgsz, args.opset)
    int8_path = quantize_int8(onnx_path)

    if args.benchmark:
        benchmark(args.weights, onnx_path, int8_path, args.imgsz, args.iters)
