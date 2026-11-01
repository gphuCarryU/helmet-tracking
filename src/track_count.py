"""Tracking (ByteTrack) + đếm phương tiện khi cắt qua vạch ảo.

Ví dụ:
    python -m src.track_count --weights weights/best.pt --source demo.mp4 \
        --line-y 0.6 --events-csv runs/track/events.csv
"""
import argparse
import csv
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def _reencode_h264(path: Path):
    """Chuyển sang H.264 để phát được trên trình duyệt (cần ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)


def process_video(weights, source, output="runs/track/output.mp4", conf=0.35,
                  imgsz=640, line_y=0.6, tracker="bytetrack.yaml",
                  events_csv=None, device=None, progress=None, classes=None):
    model = YOLO(weights)
    names = model.names

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    votes = defaultdict(Counter)   # track_id -> phiếu bầu theo lớp (chống nhấp nháy nhãn)
    last_cy = {}                   # track_id -> toạ độ y tâm ở frame trước
    counted = set()
    totals = Counter()
    events = []
    n = 0

    stream = model.track(source=str(source), stream=True, persist=True,
                         tracker=tracker, conf=conf, imgsz=imgsz,
                         device=device, classes=classes, verbose=False)
    for r in stream:
        frame = r.plot(line_width=2)
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        ly = int(h * line_y)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            for tid, c, (x1, y1, x2, y2) in zip(ids, clss, boxes):
                votes[tid][c] += 1
                cy = (y1 + y2) / 2
                prev = last_cy.get(tid)
                if prev is not None and tid not in counted and (prev - ly) * (cy - ly) < 0:
                    cls_name = names[votes[tid].most_common(1)[0][0]]
                    totals[cls_name] += 1
                    counted.add(tid)
                    events.append((n, round(n / fps, 2), tid, cls_name))
                last_cy[tid] = cy

        cv2.line(frame, (0, ly), (w, ly), (0, 255, 255), 2)
        y = 30
        for cid in sorted(names):
            cv2.putText(frame, f"{names[cid]}: {totals[names[cid]]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            y += 32
        writer.write(frame)

        n += 1
        if progress and total:
            progress(n / total)

    if writer is not None:
        writer.release()
        _reencode_h264(out_path)

    if events_csv:
        Path(events_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(events_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "time_s", "track_id", "class"])
            wr.writerows(events)

    return {"output": str(out_path), "counts": dict(totals), "frames": n, "tracks": len(votes)}


def _make_progress():
    last = {"v": -1}
    def cb(p):
        step = int(p * 10)
        if step != last["v"]:
            last["v"] = step
            print(f"Tiến độ: {step * 10}%", flush=True)
    return cb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="runs/track/output.mp4")
    p.add_argument("--events-csv", default="runs/track/events.csv")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--line-y", type=float, default=0.6, help="vị trí vạch (0-1 theo chiều cao)")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="chỉ giữ các lớp này, vd: --classes 0 1 (bỏ lớp Invalid)")
    a = p.parse_args()
    res = process_video(a.weights, a.source, a.output, a.conf, a.imgsz,
                        a.line_y, a.tracker, a.events_csv, a.device,
                        progress=_make_progress(), classes=a.classes)
    print(res)


if __name__ == "__main__":
    main()
