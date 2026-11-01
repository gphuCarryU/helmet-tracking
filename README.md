# Helmet Detection & Rider Counting (YOLOv8 + ByteTrack)

*English summary:* a YOLOv8n detector for motorbike riders with / without a helmet, trained on two merged public Roboflow datasets and applied to Vietnamese street-traffic video with ByteTrack line-crossing counting. Results, limitations and failure cases are reported as measured, including a model that initially failed on CCTV-style footage (mAP@50 ≈ 0.004) and what changed that.

## Tóm tắt

Phát hiện người đi xe có / không đội mũ bảo hiểm trên video giao thông, theo dõi bằng ByteTrack và đếm người qua một vạch ảo. Dự án gồm huấn luyện, đánh giá, đếm trên video, demo Gradio và notebook Kaggle để tái huấn luyện.

Điểm cần biết trước: model đếm **đầu người**, nên một xe chở hai người được tính là hai.

## Kết quả

**Phát hiện (YOLOv8n, ảnh vào 1280 px, train 1,02 giờ trên GPU T4)**

| Tập | Số ảnh | mAP@50 | mAP@50-95 | With Helmet AP@50 | Without Helmet AP@50 |
|---|---|---|---|---|---|
| Val gộp | 266 | 0,614 | 0,278 | 0,707 | 0,598 (88 box) |
| Test bộ cũ (ảnh đa dạng) | 69 | 0,748 | 0,419 | 0,844 | 0,653 (28 box) |
| Test Mapua (camera giám sát) | 139 | 0,542 | 0,210 | 0,725 | 0,404 (12 box) |

So sánh với model huấn luyện chỉ trên bộ dữ liệu đầu tiên (không có dữ liệu giám sát):

| Tập test | Model cũ | Model gộp dữ liệu |
|---|---|---|
| Test Mapua, mAP@50 | 0,001-0,004 | **0,542** |
| Test bộ cũ, mAP@50 | 0,840 | 0,748 |
| Test bộ cũ, Without Helmet AP@50 | 0,839 | 0,653 |

Gộp dữ liệu giúp rất nhiều trên cảnh giám sát nhưng làm giảm điểm trên bộ ảnh cũ, chủ yếu ở lớp không đội mũ (chỉ 28 box test nên số này dao động nhiều).

Huấn luyện tái lập được: chạy lại hai lần trên cùng loại GPU cho đúng cùng các số trên.

## Đếm trên video

Video thử: 13 giây, 1080p, 60 fps, camera cố định nhìn từ trên cao xuống đường phố đông xe máy. Người đi xe chạy ra xa camera. Số liệu dưới đây là số **người cắt qua vạch**, chưa phải độ chính xác (xem phần tiếp theo).

| conf | Vị trí vạch | Tổng đếm | 5 giây đầu | Số track |
|---|---|---|---|---|
| 0,25 | 0,65 | 20 | 8 | 280 |
| 0,10 | 0,55 | 25 | 7 | 264 |
| 0,10 | 0,65 | 21 | 8 | 264 |
| 0,10 | 0,75 | 16 | 10 | 264 |

Chỉ riêng việc dời vạch đã làm tổng đếm dao động từ 16 đến 25, nên không nên coi con số tổng là phép đo chính xác.

**Kiểm tra bằng đếm tay (một đoạn 5 giây, 9 người cắt qua vạch):**

| | Có đội mũ | Không đội mũ |
|---|---|---|
| Đếm tay | 8 | 1 |
| Model | 8 | 0 |

Số người có mũ khớp về tổng, nhưng mình chưa đối chiếu từng lần đếm với video, nên chưa thể nói từng lần đếm có đúng không (có hai lần đếm cùng một khung hình, có thể là người lái và người ngồi sau, và hai lần đếm cách nhau hai khung hình, có thể là một người bị đếm hai lần). **Model bỏ sót người duy nhất không đội mũ.** Đây chỉ là một mẫu rất nhỏ, không đủ để suy ra độ chính xác tổng quát.

## Quá trình và những gì học được

1. Model đầu tiên (bộ ảnh đa dạng, 629 ảnh) đạt mAP@50 = 0,84 trên tập test của chính nó, nhưng trên video giao thông chỉ đếm được 7 người trong cả video và gần như không phát hiện gì trên cảnh giám sát.
2. Nguyên nhân là chênh lệch miền dữ liệu: đầu người nhỏ (box cao khoảng 22-27 px khi thu về 640 px), ảnh nhòe và nén, trong khi dữ liệu đầu tiên là ảnh sắc nét, chụp cận.
3. Gộp thêm bộ Mapua (camera giám sát) và huấn luyện ở 1280 px. Các quyết định chưa được thử nghiệm đối chứng (ablation): giữ lớp `Invalid` làm lớp thứ ba rồi bỏ qua khi đếm (thay vì xóa), nhân ba số lần xuất hiện của ảnh có lớp không đội mũ, và dùng ảnh 1280 px.
4. Phát hiện ảnh Mapua bị ép về hình vuông 640×640 trong khi cảnh gốc là 16:9, nên đã trả về 1280×720 khi huấn luyện. Khi thử riêng với model cũ, việc này không cải thiện điểm (0,0012 so với 0,0042), nên mình không khẳng định nó có lợi. Mình giữ vì nó khớp với hình học của video.
5. Các điều thử nhưng không cho kết quả có ích: so sánh YOLOv8n với YOLOv8s trên bộ đầu tiên (kết quả đổi chiều giữa hai lần chạy ở hai máy khác nhau, nên không phân biệt được), đánh giá theo ngày/đêm (bộ dữ liệu chỉ có 2 ảnh "đêm" và 1 ảnh "đông"), và xuất ONNX (trên CPU chậm hơn PyTorch: 98,5 so với 86,8 ms mỗi ảnh, đo trên model cũ).

## Hạn chế

- Lớp không đội mũ rất hiếm: 544 box train so với 8.985 box có mũ (khoảng 1:16). Tập test chỉ có 28 và 12 box không mũ, nên số liệu của lớp này **không đáng tin**.
- Tập test Mapua cùng camera và cùng ngã tư với tập train nên điểm trên đó cao hơn thực tế. Chưa có tập kiểm tra độc lập có gán nhãn từ chính cảnh quay ở Việt Nam.
- Đánh giá đếm trên video chỉ là một đoạn 5 giây với 9 người.
- Quan sát thấy báo nhầm lặp lại: một người đứng cạnh xe đỗ bị gán nhãn không đội mũ ở nhiều khung hình (chưa cắt qua vạch nên chưa bị đếm). Chưa có bộ lọc theo vùng đường hay theo chuyển động.
- Một số ảnh `No-Helmet` của bộ Mapua có vẻ là người đi bộ, nhận xét bằng mắt từ vài chục ảnh minh họa.
- Hộp chưa khít với đầu nhỏ (mAP@50-95 chỉ 0,21-0,28).

## Cách chạy

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# lapx đã có trong requirements.txt; nếu tracking vẫn báo thiếu thư viện thì chạy: pip install lapx
```

Đặt model vào `weights/best.pt` (tải `best_merged.pt` từ mục Releases của repo, rồi đổi tên). Kiểm tra:

```bash
python -c "from ultralytics import YOLO; print(YOLO('weights/best.pt').names)"
```

Kết quả đúng: `{0: 'With Helmet', 1: 'Without Helmet', 2: 'Invalid'}`.

```bash
# Demo web (ảnh và video)
python app.py

# Đếm trên video, bỏ lớp Invalid
python -m src.track_count --weights weights/best.pt --source video.mp4 --imgsz 1280 --conf 0.1 --line-y 0.65 --classes 0 1
```

Kết quả nằm ở `runs/track/output.mp4` và `runs/track/events.csv` (mỗi dòng là một lần đếm: khung hình, giây, ID, lớp).

**Huấn luyện lại:** dùng `notebooks/helmet_kaggle_merged.ipynb` trên Kaggle (GPU T4, bật Internet, gắn secret `ROBOFLOW_API_KEY`, thêm dataset chứa project và video), bấm *Save & Run All*. Mất khoảng 1,5-2 giờ.

## Dữ liệu và giấy phép

- **Bộ 1:** Dendel Pino, "Motorcycle Helmet", Roboflow Universe (`dendel-pino-27gsx/motorcycle-helmet-fhsbn`), 629 ảnh, 2 lớp. **Giấy phép: CC BY 4.0.**
- **Bộ 2:** Mapua University, "Motorcycle Helmet Computer Vision Model", Roboflow Universe (`mapua-university-n7qm7/motorcycle-helmet-7d2gr`), **CC BY 4.0**. Đã đổi lớp (`Full-Faced`, `Half-Faced` gộp thành `With Helmet`, `No-Helmet` thành `Without Helmet`, `Invalid` giữ làm lớp thứ ba) và trả ảnh về tỉ lệ 16:9.
- **Video thử nghiệm:** "Người đi xe máy trên đường phố Sài Gòn" của Pew Nguyen, Pexels, https://www.pexels.com/vi-vn/video/thanh-ph-xe-h-i-d-ng-giao-thong-13275194/ , Pexels License (miễn phí sử dụng, không bắt buộc ghi nguồn). Không đưa video gốc vào repo.

## Sử dụng có trách nhiệm

- Model đếm đầu người, nhìn từ xa và hay nhầm với mũ lưỡi trai, khăn trùm đầu hoặc người đi bộ. Mọi quyết định xử phạt phải có người kiểm tra lại.
- Video giao thông có thể ghi hình khuôn mặt và biển số. Không đăng video gốc, và nếu đăng ảnh minh họa thì phải làm mờ hoặc hạ độ phân giải.
- Đây là dự án học tập, không phải hệ thống sẵn sàng cho thực thi pháp luật.

## Cấu trúc

```
src/train.py          huấn luyện
src/evaluate.py       đánh giá mAP / precision / recall
src/track_count.py    tracking ByteTrack + đếm qua vạch + ghi CSV
src/export_onnx.py    xuất ONNX và đo tốc độ CPU (đã thử, không nhanh hơn PyTorch)
src/slices.py         đánh giá theo điều kiện ngày/đêm (không có ý nghĩa với bộ dữ liệu này)
app.py                demo Gradio
notebooks/            notebook Kaggle: tải dữ liệu, gộp, train, đánh giá, thử vạch đếm
data/                 file cấu hình dữ liệu
```
