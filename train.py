







import os
from ultralytics import YOLO

def train_license_plate_model():







    pretrained_weights = "Train_8k_Images_v1/weights/best.pt"
    if not os.path.exists(pretrained_weights):

        pretrained_weights = "yolov8n.pt"

    print(f"[PBL4 TRAINING] Khởi tạo mô hình từ trọng số nền: {pretrained_weights}")
    model = YOLO(pretrained_weights)


    train_results = model.train(

        data="Find-license-plates-4/data.yaml",
        epochs=30,
        batch=16,
        imgsz=640,
        device=0,
        workers=8,


        box=7.5,
        cls=0.5,
        dfl=1.5,


        close_mosaic=10,
        cos_lr=False,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        momentum=0.937,
        weight_decay=0.0005,


        project="PBL4_Project",
        name="Train_Final_PBL4",
        save=True,
        plots=True,
        exist_ok=True
    )

    print("\n[PBL4 TRAINING] Huấn luyện thành công!")
    print("Trọng số tối ưu nhất được lưu tại: Train_Final_PBL4/weights/best.pt")
    print("Nhật ký chi tiết 30 Epochs lưu tại: Train_Final_PBL4/results.csv")
    return train_results

if __name__ == "__main__":
    train_license_plate_model()
