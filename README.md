# TrafficAI — Nhận diện biển số và giám sát phương tiện

Đồ án xây dựng hệ thống nhận diện phương tiện và biển số xe, lưu lịch sử phát hiện và hiển thị dữ liệu trên dashboard web.

## Sản phẩm nộp

- **Slide báo cáo:** [`presentation/Dual_AI_Bien_so_xe_V2.pptx`](presentation/Dual_AI_Bien_so_xe_V2.pptx) (bản gốc 37 slide).
- **Báo cáo PDF:** [`docs/Cao_Tien_Dung_NCKH_DA.pdf`](docs/Cao_Tien_Dung_NCKH_DA.pdf).
- **Mã nguồn:** ứng dụng ở `main.py`, `web_app.py`, `src/`, `templates/`, `static/`.
- **Kết quả huấn luyện:** biểu đồ và bảng trong [`Train_Final_PBL4/results_and_charts/`](Train_Final_PBL4/results_and_charts/).
- **Dữ liệu minh họa:** [`dataset_characters/`](dataset_characters/) có 2.520 ảnh ký tự tổng hợp. Đây không phải bộ ảnh biển số giao thông thật. Link bộ dữ liệu thật (nếu được phép chia sẻ) cần bổ sung riêng: **[DÁN LINK DỮ LIỆU THẬT Ở ĐÂY]**.

## Chức năng chính

- Phát hiện và theo dõi phương tiện bằng YOLO.
- Nhận diện ký tự biển số bằng OCR.
- Dashboard FastAPI để tra cứu thống kê và lịch sử nhận diện.
- Lưu các sự kiện nhận diện vào SQLite.

## Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

## Chạy ứng dụng

Mở dashboard:

```bash
python web_app.py
```

Mở `http://127.0.0.1:8001` trên trình duyệt.

Chạy pipeline xử lý video:

```bash
python main.py
```

Pipeline cần file video đầu vào và model YOLO tương ứng. Không có video/bộ dữ liệu cá nhân trong repo; xem link dữ liệu ở phần **Sản phẩm nộp**.

## Mô-đun CNN ký tự

Repo có `src/cnn_model.py`, trọng số `models/plate_char_cnn.pth` và script `train_cnn.py`. Script tự sinh dữ liệu tổng hợp: 36 × 350 = 12.600 mẫu train và 36 × 70 = 2.520 mẫu validation; không huấn luyện trực tiếp trên ảnh biển số thật. `dataset_characters/` là một bản xuất validation tổng hợp riêng, không được `train_cnn.py` đọc. Các kết quả này không chứng minh độ chính xác trên ảnh giao thông thực.

`benchmark_real_plates.py` cần ảnh crop và nhãn phù hợp trong `static/crops/`; thư mục này không được đưa lên repo vì có thể chứa biển số nhận dạng được. Mô-đun CNN hiện được lưu riêng, chưa nối vào pipeline web hiện tại.

## Kết quả

Xem `Train_Final_PBL4/results_and_charts/` để tham khảo `results.csv`, các biểu đồ huấn luyện và ảnh minh họa inference. Ghi rõ nguồn và cách tạo từng kết quả khi dùng trong báo cáo.

## Cấu trúc

```text
.
├── docs/                         # Báo cáo PDF
├── presentation/                 # Slide gốc và kịch bản thuyết trình
├── src/                          # Engine, database, models, security shield
├── static/                       # CSS/JS/dashboard assets
├── templates/                    # Giao diện web
├── dataset_characters/           # Ảnh ký tự tổng hợp (validation minh họa)
├── models/plate_char_cnn.pth     # Trọng số CNN ký tự từ bản lưu
├── Train_Final_PBL4/
│   ├── results_and_charts/        # Bảng và hình kết quả
│   └── weights/best.pt            # Model đã huấn luyện
├── main.py
├── web_app.py
└── requirements*.txt
```

## Lưu ý khi chia sẻ

Repo loại trừ môi trường ảo, database cục bộ, video, ảnh crop biển số, dữ liệu sao lưu và log. Trước khi công khai repo, hãy kiểm tra lại nội dung bộ dữ liệu và quyền chia sẻ link dữ liệu.
