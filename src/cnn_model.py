import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(CHAR_LIST)}
IDX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHAR_LIST)}

class PlateCharCNN(nn.Module):






    def __init__(self, num_classes=36):
        super(PlateCharCNN, self).__init__()


        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)


        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.2)


        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)


        self.flatten_dim = 128 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.35)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)


        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)


        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)


        x = x.view(-1, self.flatten_dim)
        x = self.drop_fc(F.relu(self.bn_fc(self.fc1(x))))
        out = self.fc2(x)
        return out


class CharacterSegmenter:







    @staticmethod
    def preprocess_plate(plate_bgr):
        if plate_bgr is None or plate_bgr.size == 0:
            return None, None

        h, w = plate_bgr.shape[:2]

        target_h = 100
        target_w = int(w * (target_h / float(max(h, 1))))
        resized = cv2.resize(plate_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)


        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return resized, binary

    @classmethod
    def segment_characters(cls, plate_bgr):
        resized, binary = cls.preprocess_plate(plate_bgr)
        if binary is None:
            return []

        h, w = binary.shape
        is_two_lines = (w / float(h)) < 2.3


        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_candidates = []
        min_h = int(h * 0.25)
        max_h = int(h * 0.90)
        min_w = int(w * 0.03)
        max_w = int(w * 0.35)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(ch)
            area = cw * ch
            solidity = cv2.contourArea(cnt) / float(area + 1e-5)


            if (min_h <= ch <= max_h) and (min_w <= cw <= max_w) and (0.15 <= aspect_ratio <= 1.1) and (solidity > 0.15):

                char_crop = binary[y:y+ch, x:x+cw]


                pad = max(int(ch * 0.12), 3)
                char_padded = cv2.copyMakeBorder(char_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


                char_normalized = cv2.resize(char_padded, (32, 32), interpolation=cv2.INTER_AREA)

                char_candidates.append({
                    "box": (x, y, cw, ch),
                    "cy": y + ch / 2.0,
                    "cx": x + cw / 2.0,
                    "img": char_normalized
                })

        if not char_candidates:
            return []


        if is_two_lines:

            all_cy = [c["cy"] for c in char_candidates]
            mid_y = np.median(all_cy)
            top_line = [c for c in char_candidates if c["cy"] <= mid_y]
            bottom_line = [c for c in char_candidates if c["cy"] > mid_y]

            top_line.sort(key=lambda c: c["cx"])
            bottom_line.sort(key=lambda c: c["cx"])
            sorted_chars = top_line + bottom_line
        else:

            sorted_chars = sorted(char_candidates, key=lambda c: c["cx"])

        return sorted_chars


class PlateCNNEngine:






    _instance = None
    _model = None
    _device = None

    def __init__(self, weights_path="models/plate_char_cnn.pth"):
        if not os.path.isabs(weights_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_dir, weights_path)
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlateCharCNN(num_classes=36).to(self.device)
        self.model.eval()
        self.is_ready = False
        self.load_weights()

    def load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_ready = True
                print(f"[CNN ENGINE] Successfully loaded CNN character weights from: {self.weights_path}")
            except Exception as e:
                print(f"[CNN ENGINE] Warning: Failed to load weights ({e}). Initialized fresh CNN architecture.")
        else:
            print(f"[CNN ENGINE] Notice: Weights file {self.weights_path} not found. Running with base architecture.")

    def predict_char(self, char_img_32x32):



        tensor = torch.from_numpy(char_img_32x32).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        char = IDX_TO_CHAR[pred_idx.item()]
        return char, conf.item()

    def predict_plate_text(self, plate_crop):




        chars_data = CharacterSegmenter.segment_characters(plate_crop)
        if not chars_data or len(chars_data) < 4:
            return None, 0.0, []

        predicted_chars = []
        confidences = []
        details = []

        for item in chars_data:
            ch, conf = self.predict_char(item["img"])
            predicted_chars.append(ch)
            confidences.append(conf)
            details.append({"char": ch, "conf": round(conf, 3), "box": item["box"]})

        raw_text = "".join(predicted_chars)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return raw_text, avg_conf, details
