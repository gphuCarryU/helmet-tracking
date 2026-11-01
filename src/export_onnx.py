"""Xuất ONNX và so sánh tốc độ PyTorch vs ONNX trên CPU (end-to-end, batch 1).

Ví dụ:
    python -m src.export_onnx --weights runs/train/baseline_n/weights/best.pt \
        --bench-dir data/dataset/images/test
"""
import argparse
import time
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def bench(model, imgs, imgsz):
    for im in imgs[:5]:  # warm-up
        model.predict(str(im), imgsz=imgsz, device="cpu", verbose=False)
    t0 = time.perf_counter()
    for im in imgs:
        model.predict(str(im), imgsz=imgsz, device="cpu", verbose=False)
    return (time.perf_counter() - t0) / len(imgs) * 1000  # ms/ảnh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--bench-n", type=int, default=100)
    a = p.parse_args()

    pt_path = Path(a.weights)
    onnx_path = Path(YOLO(a.weights).export(format="onnx", imgsz=a.imgsz,
                                            simplify=True, dynamic=False))
    imgs = sorted(p_ for p_ in Path(a.bench_dir).iterdir()
                  if p_.suffix.lower() in IMG_EXT)[: a.bench_n]
    if not imgs:
        raise SystemExit("Không có ảnh để benchmark")

    pt_ms = bench(YOLO(str(pt_path)), imgs, a.imgsz)
    onnx_ms = bench(YOLO(str(onnx_path), task="detect"), imgs, a.imgsz)

    print(f"\nBenchmark CPU, {len(imgs)} ảnh, imgsz={a.imgsz}")
    print("| Định dạng | Dung lượng (MB) | ms/ảnh | FPS |")
    print("|---|---|---|---|")
    print(f"| PyTorch (.pt) | {pt_path.stat().st_size / 1e6:.1f} | {pt_ms:.1f} | {1000 / pt_ms:.1f} |")
    print(f"| ONNX | {onnx_path.stat().st_size / 1e6:.1f} | {onnx_ms:.1f} | {1000 / onnx_ms:.1f} |")
    print(f"\nFile ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
