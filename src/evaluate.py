"""Đánh giá model trên tập test: mAP, precision, recall từng lớp và đo FPS.

Ví dụ:
    python -m src.evaluate --weights runs/train/baseline_n/weights/best.pt \
        --bench-dir data/dataset/images/test
"""
import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--bench-dir", default=None, help="thư mục ảnh để đo FPS")
    p.add_argument("--bench-n", type=int, default=100)
    args = p.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  project=str(Path("runs/eval").resolve()), name=Path(args.weights).parts[-3])

    print("\n=== KẾT QUẢ TỔNG ===")
    print(f"mAP@50     : {m.box.map50:.4f}")
    print(f"mAP@50-95  : {m.box.map:.4f}")
    print(f"Precision  : {m.box.mp:.4f}")
    print(f"Recall     : {m.box.mr:.4f}")

    print("\n=== TỪNG LỚP ===")
    for i, cls_id in enumerate(m.box.ap_class_index):
        pr, rc, ap50, ap = m.box.class_result(i)
        print(f"{model.names[int(cls_id)]:<12} P={pr:.3f} R={rc:.3f} "
              f"AP50={ap50:.3f} AP50-95={ap:.3f}")

    if args.bench_dir:
        imgs = sorted(Path(args.bench_dir).glob("*.*"))[: args.bench_n]
        if imgs:
            for im in imgs[:5]:  # warm-up
                model.predict(str(im), imgsz=args.imgsz, verbose=False)
            t0 = time.perf_counter()
            for im in imgs:
                model.predict(str(im), imgsz=args.imgsz, verbose=False)
            dt = time.perf_counter() - t0
            print(f"\nFPS (end-to-end, {len(imgs)} ảnh): {len(imgs) / dt:.1f}")


if __name__ == "__main__":
    main()
