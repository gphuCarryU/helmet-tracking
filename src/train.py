"""Fine-tune YOLO cho bài toán phát hiện mũ bảo hiểm.

Ví dụ:
    python -m src.train --model yolov8n.pt --name baseline_n
    python -m src.train --model yolov8s.pt --imgsz 800 --name yolov8s_800
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="baseline")
    p.add_argument("--device", default=None, help="vd: 0 hoặc cpu")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(Path("runs/train").resolve()),
        name=args.name,
        device=args.device,
        seed=42,
        plots=True,
        # augmentation phù hợp ảnh giao thông (đổi thử để làm ablation)
        hsv_v=0.4,
        mosaic=1.0,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()
