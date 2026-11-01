"""Đánh giá model theo từng lát cắt điều kiện (ban ngày/đêm, thưa/đông, ...).

Bước 1 - tự sinh file gắn nhãn điều kiện (sau đó nên xem lại và sửa tay):
    python -m src.slices make-template --img-dir data/dataset/images/test

Bước 2 - đánh giá từng lát cắt:
    python -m src.slices evaluate --weights runs/train/baseline_n/weights/best.pt

Bạn có thể thêm cột mới vào data/slices.csv (vd: weather, camera_angle),
mỗi cột sẽ thành một nhóm lát cắt.
"""
import argparse
import csv
import tempfile
from pathlib import Path

import cv2
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path(img: Path) -> Path:
    """.../images/test/a.jpg -> .../labels/test/a.txt (quy ước YOLO)."""
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def make_template(img_dir, out_csv, night_thr, crowd_thr):
    img_dir = Path(img_dir)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"Không thấy ảnh trong {img_dir}")
    rows = []
    for p in imgs:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        lighting = "night" if float(gray.mean()) < night_thr else "day"
        lp = label_path(p.resolve())
        n_obj = len([l for l in lp.read_text().splitlines() if l.strip()]) if lp.exists() else 0
        density = "crowded" if n_obj >= crowd_thr else "sparse"
        rows.append({"image": p.name, "lighting": lighting, "density": density})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "lighting", "density"])
        w.writeheader()
        w.writerows(rows)
    n_night = sum(r["lighting"] == "night" for r in rows)
    n_crowd = sum(r["density"] == "crowded" for r in rows)
    print(f"Đã ghi {len(rows)} dòng vào {out_csv} "
          f"(night={n_night}, crowded={n_crowd}).")
    print("Nhãn tự động dựa trên độ sáng trung bình và số đối tượng/ảnh; "
          "hãy mở file ra kiểm tra lại vài ảnh.")


def eval_subset(model, cfg, paths, imgsz, workdir, tag):
    lst = workdir / f"{tag}.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in paths))
    sub = dict(cfg)
    sub.pop("path", None)
    for k in ("train", "val", "test"):
        sub[k] = str(lst)
    yml = workdir / f"{tag}.yaml"
    yml.write_text(yaml.safe_dump(sub, allow_unicode=True))
    m = model.val(data=str(yml), split="test", imgsz=imgsz, plots=False,
                  verbose=False, project=str(workdir / "val"), name=tag,
                  exist_ok=True)
    return m.box.map50, m.box.map, m.box.mp, m.box.mr


def evaluate(a):
    from ultralytics import YOLO  # import trễ để make-template chạy không cần ultralytics

    cfg = yaml.safe_load(open(a.data))
    if a.img_dir:
        test_dir = Path(a.img_dir)
    else:
        base = Path(cfg.get("path") or Path(a.data).resolve().parent)
        test_dir = Path(cfg["test"])
        if not test_dir.is_absolute():
            test_dir = base / test_dir

    rows = list(csv.DictReader(open(a.slices)))
    cols = [c for c in rows[0] if c != "image"]
    model = YOLO(a.weights)

    out = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        everything = [test_dir / r["image"] for r in rows]
        groups = [("all", "all", everything)]
        for col in cols:
            for v in sorted({r[col] for r in rows if r[col]}):
                groups.append((col, v, [test_dir / r["image"] for r in rows if r[col] == v]))

        for col, v, paths in groups:
            if len(paths) < a.min_images and col != "all":
                out.append((col, v, len(paths), None))
                continue
            res = eval_subset(model, cfg, paths, a.imgsz, td, f"{col}_{v}")
            out.append((col, v, len(paths), res))

    print("\n| Điều kiện | Giá trị | Số ảnh | mAP@50 | mAP@50-95 | Precision | Recall |")
    print("|---|---|---|---|---|---|---|")
    for col, v, n, res in out:
        if res is None:
            print(f"| {col} | {v} | {n} | (ít hơn {a.min_images} ảnh, bỏ qua) | | | |")
        else:
            print(f"| {col} | {v} | {n} | {res[0]:.3f} | {res[1]:.3f} | {res[2]:.3f} | {res[3]:.3f} |")

    out_csv = Path("runs/eval/slices.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "value", "n_images", "map50", "map50_95", "precision", "recall"])
        for col, v, n, res in out:
            w.writerow([col, v, n] + ([f"{x:.4f}" for x in res] if res else ["", "", "", ""]))
    print(f"\nĐã lưu: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("make-template")
    t.add_argument("--img-dir", required=True)
    t.add_argument("--out", default="data/slices.csv")
    t.add_argument("--night-thr", type=float, default=90.0,
                   help="độ sáng xám trung bình (0-255) dưới ngưỡng này => night")
    t.add_argument("--crowd-thr", type=int, default=5,
                   help="số đối tượng/ảnh từ ngưỡng này trở lên => crowded")

    e = sp.add_parser("evaluate")
    e.add_argument("--weights", required=True)
    e.add_argument("--data", default="data/data.yaml")
    e.add_argument("--slices", default="data/slices.csv")
    e.add_argument("--img-dir", default=None)
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--min-images", type=int, default=20,
                   help="lát cắt ít ảnh hơn mức này thì bỏ qua (số liệu không đáng tin)")

    a = p.parse_args()
    if a.cmd == "make-template":
        make_template(a.img_dir, a.out, a.night_thr, a.crowd_thr)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
