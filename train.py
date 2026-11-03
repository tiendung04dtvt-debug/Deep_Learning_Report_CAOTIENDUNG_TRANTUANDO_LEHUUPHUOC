"""
TRAINING SCRIPT: YOLOv8 License Plate Detection (Fine-Tuning)
Dự án: Hệ thống giám sát giao thông & nhận diện biển số xe thời gian thực (PBL4)
Sinh viên thực hiện: Cao Tiến Dũng, Trần Tuấn Đô, Lê Hữu Phước
Giảng viên hướng dẫn: TS. Trần Thị Minh Hạnh
Bộ môn: Kỹ thuật Viễn thông - Trường Đại học Bách Khoa, ĐH Đà Nẵng
"""

import os
from ultralytics import YOLO

def train_license_plate_model():
    """
    Quy trình huấn luyện tinh chỉnh (Fine-tuning) mô hình YOLOv8n định vị biển số xe.
    Mô hình được huấn luyện trên môi trường điện toán đám mây Google Colab GPU (Tesla T4)
    để tối ưu hóa tốc độ tính toán gradient.
    """
    # 1. Khởi tạo mô hình nền tảng (Transfer Learning)
    # Khởi tạo từ checkpoint pre-train đã học trên 8.000 ảnh phương tiện
    pretrained_weights = "Train_8k_Images_v1/weights/best.pt"
    if not os.path.exists(pretrained_weights):
        # Fallback về trọng số chuẩn YOLOv8n nếu chạy kiểm tra cục bộ
        pretrained_weights = "yolov8n.pt"

    print(f"[PBL4 TRAINING] Khởi tạo mô hình từ trọng số nền: {pretrained_weights}")
    model = YOLO(pretrained_weights)

    # 2. Thực thi huấn luyện với toàn bộ siêu tham số khớp cấu hình Train_Final_PBL4/args.yaml
    train_results = model.train(
        # --- DỮ LIỆU ĐẦU VÀO ---
        data="Find-license-plates-4/data.yaml",  # Cấu hình tập dữ liệu 2.500 ảnh biển số xe Việt Nam
        epochs=30,                              # 30 Epochs fine-tuning
        batch=16,                               # Batch size = 16
        imgsz=640,                              # Độ phân giải ảnh 640x640 RGB
        device=0,                               # Tăng tốc phần cứng GPU CUDA (device '0')
        workers=8,                              # Số luồng đọc dữ liệu song song

        # --- CẤU HÌNH TRỌNG SỐ HÀM MẤT MÁT (LOSS FUNCTIONS) ---
        box=7.5,                                # Box Loss CIoU: Trọng số cao nhất để định vị chính xác khung bao
        cls=0.5,                                # Class Loss BCE: Trọng số vừa đủ cho bài toán đơn lớp (license plate)
        dfl=1.5,                                # Distribution Focal Loss: Tối ưu phân phối viền mờ của tọa độ

        # --- CHIẾN LƯỢC LỊCH TRÌNH HỌC & TĂNG CƯỜNG DỮ LIỆU ---
        close_mosaic=10,                        # Tắt kỹ thuật ghép ảnh Mosaic ở 10 Epoch cuối để tinh chỉnh trên phân phối thực
        cos_lr=False,                           # Lịch trình suy giảm tốc độ học tuyến tính (Linear Decay)
        lr0=0.01,                               # Learning rate khởi điểm
        lrf=0.01,                               # Tỷ lệ suy giảm LR cuối cùng (Final LR = 0.01 * lr0)
        warmup_epochs=3.0,                      # Khởi động tăng dần LR ở giai đoạn đầu để ổn định gradient
        momentum=0.937,                         # Momentum tối ưu hóa
        weight_decay=0.0005,                    # Hệ số phạt chống overfitting

        # --- ĐẦU RA & NHẬT KÝ (LOGGING) ---
        project="PBL4_Project",                 # Tên thư mục dự án
        name="Train_Final_PBL4",                # Tên phiên huấn luyện
        save=True,                              # Tự động lưu checkpoint best.pt và last.pt
        plots=True,                             # Tự động vẽ đồ thị metrics/loss và ma trận nhầm lẫn
        exist_ok=True
    )

    print("\n[PBL4 TRAINING] Huấn luyện thành công!")
    print("Trọng số tối ưu nhất được lưu tại: Train_Final_PBL4/weights/best.pt")
    print("Nhật ký chi tiết 30 Epochs lưu tại: Train_Final_PBL4/results.csv")
    return train_results

if __name__ == "__main__":
    train_license_plate_model()
