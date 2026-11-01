"""Demo Gradio: phát hiện mũ bảo hiểm trên ảnh và đếm trên video."""
import json
import os
import tempfile

import gradio as gr
import numpy as np
from ultralytics import YOLO

from src.track_count import process_video

WEIGHTS = os.getenv("WEIGHTS", "weights/best.pt")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS)
    return _model


def detect_image(img, conf):
    if img is None:
        return None
    bgr = np.ascontiguousarray(img[..., ::-1])
    r = get_model().predict(bgr, conf=conf, imgsz=1280, classes=[0, 1], verbose=False)[0]
    return np.ascontiguousarray(r.plot()[..., ::-1])


def detect_video(video, conf, line_y, progress=gr.Progress()):
    if video is None:
        return None, "{}"
    out = os.path.join(tempfile.mkdtemp(), "result.mp4")
    res = process_video(WEIGHTS, video, out, conf=conf, imgsz=1280, line_y=line_y,
                        classes=[0, 1], progress=lambda p: progress(p))
    return res["output"], json.dumps(res["counts"], indent=2, ensure_ascii=False)


with gr.Blocks(title="Helmet Detection & Counting") as demo:
    gr.Markdown("# Phát hiện mũ bảo hiểm & đếm phương tiện (YOLO + ByteTrack)")
    with gr.Tab("Ảnh"):
        i = gr.Image(type="numpy", label="Ảnh đầu vào")
        c1 = gr.Slider(0.1, 0.9, 0.25, label="Ngưỡng confidence")
        o = gr.Image(label="Kết quả")
        gr.Button("Phát hiện").click(detect_image, [i, c1], o)
    with gr.Tab("Video"):
        v = gr.Video(label="Video đầu vào")
        c2 = gr.Slider(0.1, 0.9, 0.25, label="Ngưỡng confidence")
        ly = gr.Slider(0.1, 0.9, 0.65, label="Vị trí vạch đếm")
        ov = gr.Video(label="Kết quả")
        cnt = gr.Code(label="Số lượng đếm được", language="json")
        gr.Button("Xử lý video").click(detect_video, [v, c2, ly], [ov, cnt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
