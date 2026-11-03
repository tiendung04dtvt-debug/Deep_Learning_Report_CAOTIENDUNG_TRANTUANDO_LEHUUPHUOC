import os
import sys
import re
import time
import numpy as np
from difflib import SequenceMatcher

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import torch
from src.cnn_model import PlateCNNEngine
from src.engine import get_easyocr_reader

def cv2_imread_utf8(path):
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return None

def clean_plate(s):
    if not s:
        return ""
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())

def char_accuracy(pred, gt):
    pred = clean_plate(pred)
    gt = clean_plate(gt)
    if not gt:
        return 0.0
    if pred == gt:
        return 1.0
    matcher = SequenceMatcher(None, pred, gt)
    return matcher.ratio()

def format_plate_vietnam(text):
    if not text:
        return "", False
    clean = clean_plate(text)
    if len(clean) >= 7 and re.match(r'^\d{2}[A-Z]', clean):
        return clean, True
    return clean, False

def run_benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    crop_dir = os.path.join(base_dir, "static", "crops")
    all_files = sorted([f for f in os.listdir(crop_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    known_gt_map = {
        '292Y303658': '29Y303658',
        '29Y303658': '29Y303658',
        '51B60174': '51B60174',
        '51F13619': '51F13619',
        '51H04964': '51H04964',
        '51H27087': '51H27087',
        '5479999': '54L19999',
        '54L19999': '54L19999',
        '60A02613': '60A02613',
        '60K01109': '60K01109',
        '61A43854': '61A43854',
        '72A47115': '72A47115',
        '72A52976': '72A52976',
        '72F00075': '72F00075',
        '74AYJZMH': '74A11258',
        '76E00154': '76E00154',
        '84E00181': '84E00181'
    }

    test_samples = []
    for f in all_files:
        prefix = f.split('_')[0]
        gt = known_gt_map.get(prefix)
        if gt:
            img_path = os.path.join(crop_dir, f)
            img = cv2_imread_utf8(img_path)
            if img is not None and img.shape[0] > 10 and img.shape[1] > 20:
                test_samples.append((f, img, gt))

    print(f"[BENCHMARK] Loaded {len(test_samples)} valid real-world plate crops from video stream.")
    
    # Initialize engines
    print("[BENCHMARK] Initializing Custom CNN Engine...")
    weights_path = os.path.join(base_dir, "models", "plate_char_cnn.pth")
    cnn_engine = PlateCNNEngine(weights_path)
    
    # Verify weights loaded
    if not cnn_engine.is_ready:
        print("[BENCHMARK] Reloading CNN weights via binary buffer...")
        try:
            with open(weights_path, "rb") as f:
                state_dict = torch.load(f, map_location=cnn_engine.device)
                cnn_engine.model.load_state_dict(state_dict)
                cnn_engine.model.eval()
                cnn_engine.is_ready = True
                print("[BENCHMARK] CNN weights successfully loaded!")
        except Exception as err:
            print(f"[BENCHMARK] CNN load err: {err}")

    print("[BENCHMARK] Initializing EasyOCR Reader (CUDA)...")
    easyocr_reader = get_easyocr_reader()

    ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    cnn_plate_correct = 0
    cnn_char_acc_sum = 0.0
    cnn_times = []
    
    ocr_plate_correct = 0
    ocr_char_acc_sum = 0.0
    ocr_times = []
    
    hybrid_plate_correct = 0
    hybrid_char_acc_sum = 0.0
    hybrid_times = []
    
    failure_cases = []

    for idx, (filename, img, gt) in enumerate(test_samples):
        h, w = img.shape[:2]
        aspect_ratio = round(w / float(max(h, 1)), 2)
        
        # --- 1. Custom CNN Alone ---
        t0 = time.perf_counter()
        c_text, c_conf, _ = cnn_engine.predict_plate_text(img)
        t_cnn = (time.perf_counter() - t0) * 1000
        cnn_times.append(t_cnn)
        pred_cnn = clean_plate(c_text) if c_text else ""
        if pred_cnn == gt:
            cnn_plate_correct += 1
        cnn_char_acc_sum += char_accuracy(pred_cnn, gt)

        # --- 2. EasyOCR Alone ---
        t0 = time.perf_counter()
        ocr_raw = ""
        try:
            enhanced = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if w < 160 or h < 60:
                enhanced = cv2.resize(enhanced, (max(w * 2, 180), max(h * 2, 70)), interpolation=cv2.INTER_CUBIC)
            eh, ew = enhanced.shape[:2]
            if aspect_ratio < 2.3:
                res = easyocr_reader.readtext(enhanced, allowlist=ALLOWLIST, detail=0)
                ocr_raw = "".join(res).strip()
            else:
                res = easyocr_reader.recognize(enhanced, horizontal_list=[[0, ew, 0, eh]], free_list=[], detail=0, allowlist=ALLOWLIST)
                ocr_raw = "".join(res).strip()
                if len(ocr_raw) < 5:
                    res2 = easyocr_reader.readtext(enhanced, allowlist=ALLOWLIST, detail=0)
                    ocr_raw = "".join(res2).strip()
        except Exception:
            ocr_raw = ""
        t_ocr = (time.perf_counter() - t0) * 1000
        ocr_times.append(t_ocr)
        pred_ocr = clean_plate(ocr_raw)
        if pred_ocr == gt:
            ocr_plate_correct += 1
        ocr_char_acc_sum += char_accuracy(pred_ocr, gt)

        # --- 3. Hybrid Fusion ---
        t0 = time.perf_counter()
        hybrid_raw = ""
        # Hybrid decision: Prioritize CNN when confidence >= 0.85 and length >= 7
        if pred_cnn and len(pred_cnn) >= 7 and c_conf >= 0.85:
            hybrid_raw = pred_cnn
        elif pred_ocr and len(pred_ocr) >= 7 and re.match(r'^\d{2}[A-Z]', pred_ocr):
            hybrid_raw = pred_ocr
        elif pred_cnn and len(pred_cnn) >= 6:
            hybrid_raw = pred_cnn
        else:
            hybrid_raw = pred_ocr if pred_ocr else pred_cnn
        
        formatted, is_trusted = format_plate_vietnam(hybrid_raw)
        pred_hybrid = clean_plate(formatted) if formatted else hybrid_raw
        t_hybrid = (time.perf_counter() - t0) * 1000 + (t_cnn if hybrid_raw == pred_cnn else (t_cnn + t_ocr))
        hybrid_times.append(t_hybrid)
        
        if pred_hybrid == gt:
            hybrid_plate_correct += 1
        else:
            if len(failure_cases) < 15:
                failure_cases.append({
                    "file": filename,
                    "gt": gt,
                    "cnn": pred_cnn,
                    "ocr": pred_ocr,
                    "hybrid": pred_hybrid,
                    "aspect_ratio": aspect_ratio
                })
        hybrid_char_acc_sum += char_accuracy(pred_hybrid, gt)

    n = len(test_samples)
    print("\n" + "="*70)
    print("                  KET QUA THUC NGHIEM TREN TAP ANH THAT")
    print("="*70)
    print(f"Tong so anh bien so that kiem thu: {n} anh (tu luong video camera)")
    print("-" * 70)
    print(f"1. Custom Deep CNN (PlateCharCNN):")
    print(f"   - Ti le dung ca bien (Full Plate Accuracy): {cnn_plate_correct}/{n} ({cnn_plate_correct/n*100:.2f}%)")
    print(f"   - Ti le dung ky tu (Char Accuracy):         {cnn_char_acc_sum/n*100:.2f}%")
    print(f"   - Do tre trung binh:                        {np.mean(cnn_times):.2f} ms")
    print("-" * 70)
    print(f"2. EasyOCR (Nguyen ban):")
    print(f"   - Ti le dung ca bien (Full Plate Accuracy): {ocr_plate_correct}/{n} ({ocr_plate_correct/n*100:.2f}%)")
    print(f"   - Ti le dung ky tu (Char Accuracy):         {ocr_char_acc_sum/n*100:.2f}%")
    print(f"   - Do tre trung binh:                        {np.mean(ocr_times):.2f} ms")
    print("-" * 70)
    print(f"3. Hybrid Fusion (De xuat):")
    print(f"   - Ti le dung ca bien (Full Plate Accuracy): {hybrid_plate_correct}/{n} ({hybrid_plate_correct/n*100:.2f}%)")
    print(f"   - Ti le dung ky tu (Char Accuracy):         {hybrid_char_acc_sum/n*100:.2f}%")
    print(f"   - Do tre trung binh:                        {np.mean(hybrid_times):.2f} ms")
    print("="*70)
    print(f"\nSo luong truong hop that bai (Failure cases): {len(failure_cases)}")
    for fc in failure_cases[:8]:
        print(f"File: {fc['file']} | GT: {fc['gt']} | CNN: {fc['cnn']} | OCR: {fc['ocr']} | Hybrid: {fc['hybrid']}")

if __name__ == "__main__":
    run_benchmark()
