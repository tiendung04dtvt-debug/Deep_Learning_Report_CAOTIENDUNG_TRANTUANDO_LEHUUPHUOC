






import os
import sys
import io
import random


if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.cnn_model import PlateCharCNN, CHAR_LIST, CHAR_TO_IDX

class SyntheticLicensePlateDataset(Dataset):








    def __init__(self, samples_per_class=400, transform=True):
        self.samples_per_class = samples_per_class
        self.transform = transform
        self.data = []
        self.labels = []
        self._generate_dataset()

    def _generate_dataset(self):
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL
        ]

        for char_idx, char in enumerate(CHAR_LIST):
            for _ in range(self.samples_per_class):

                canvas = np.zeros((64, 64), dtype=np.uint8)
                font = random.choice(fonts)
                scale = random.uniform(1.2, 1.7)
                thickness = random.choice([2, 3])


                (text_w, text_h), baseline = cv2.getTextSize(char, font, scale, thickness)
                org_x = max(2, (64 - text_w) // 2 + random.randint(-4, 4))
                org_y = max(text_h + 2, (64 + text_h) // 2 + random.randint(-4, 4))

                cv2.putText(canvas, char, (org_x, org_y), font, scale, 255, thickness, cv2.LINE_AA)

                if self.transform:

                    angle = random.uniform(-12, 12)
                    M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                    canvas = cv2.warpAffine(canvas, M, (64, 64), flags=cv2.INTER_CUBIC, borderValue=0)


                    if random.random() > 0.4:
                        noise = np.random.randint(0, 35, (64, 64), dtype=np.uint8)
                        canvas = cv2.add(canvas, noise)


                    if random.random() > 0.5:
                        k = random.choice([3, 5])
                        canvas = cv2.GaussianBlur(canvas, (k, k), 0)


                img_32x32 = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)

                self.data.append(img_32x32)
                self.labels.append(char_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]


        tensor_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return tensor_img, label


def train_plate_cnn(epochs=12, batch_size=64, lr=0.001):
    print("================================================================")
    print("  BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH CUSTOM DEEP CNN NHẬN DIỆN KÝ TỰ BIỂN SỐ")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[DEVICE] Tăng tốc GPU chuyên sâu: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print(f"[DEVICE] Sử dụng thiết bị tính toán: {device}")


    print("[1/4] Đang khởi tạo tập dữ liệu ký tự biển số (36 classes)...")
    train_dataset = SyntheticLicensePlateDataset(samples_per_class=350, transform=True)
    val_dataset = SyntheticLicensePlateDataset(samples_per_class=70, transform=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"      -> Tổng số mẫu huấn luyện (Train): {len(train_dataset)}")
    print(f"      -> Tổng số mẫu kiểm thử (Val):     {len(val_dataset)}")


    print("[2/4] Khởi tạo kiến trúc PlateCharCNN (3 khối tích chập Conv-BN-ReLU-MaxPool)...")
    model = PlateCharCNN(num_classes=36).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }


    print(f"[3/4] Bắt đầu quá trình huấn luyện qua {epochs} Epochs...")
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    weights_path = "models/plate_char_cnn.pth"

    for epoch in range(1, epochs + 1):

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0


        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), weights_path)

    print(f"\n[THÀNH CÔNG] Đã lưu trọng số mô hình tốt nhất ({best_val_acc:.2f}%) vào: {weights_path}")


    print("[4/4] Đang vẽ biểu đồ huấn luyện đánh giá (Loss & Accuracy Curves)...")
    os.makedirs("static/reports", exist_ok=True)
    report_img_path = "static/reports/cnn_training_metrics.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


    ax1.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="#ef4444", lw=2, marker='o')
    ax1.plot(range(1, epochs + 1), history["val_loss"], label="Validation Loss", color="#3b82f6", lw=2, linestyle='--', marker='s')
    ax1.set_title("Biểu đồ Hàm mất mát (Cross-Entropy Loss)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)


    ax2.plot(range(1, epochs + 1), history["train_acc"], label="Train Accuracy", color="#10b981", lw=2, marker='o')
    ax2.plot(range(1, epochs + 1), history["val_acc"], label="Validation Accuracy", color="#8b5cf6", lw=2, linestyle='--', marker='s')
    ax2.set_title("Biểu đồ Độ chính xác phân loại ký tự (Accuracy %)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Độ chính xác (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(report_img_path, dpi=200)
    plt.close()
    print(f"      -> Biểu đồ đã được lưu thành công tại: {report_img_path}")
    print("================================================================")
    print("  QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH CNN HOÀN TẤT XUẤT SẮC!")
    print("================================================================")

if __name__ == "__main__":
    train_plate_cnn(epochs=12, batch_size=64, lr=0.001)
