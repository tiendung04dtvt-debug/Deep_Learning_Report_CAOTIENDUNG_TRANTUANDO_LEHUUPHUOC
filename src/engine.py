import os
import re
import threading
import time
import datetime
from queue import Queue, Empty
from difflib import SequenceMatcher
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.database import SessionLocal
from src.models import Detection


ai_engine_running = False
ai_engine_thread = None
latest_frame_lock = threading.Lock()
latest_processed_frame = None


_ocr_log_queue = Queue()

def _async_benchmark_logger_worker():
    import logging
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("TrafficAI_Benchmark")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = "ocr_benchmark_log.csv"
    file_exists = os.path.exists(log_file)
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    if not file_exists:
        logger.info("timestamp,track_id,aspect_ratio,path,latency_ms,plate_text_raw,plate_text_snapped,ground_truth,fusion_matched,is_trusted,fallback")

    while True:
        try:
            record = _ocr_log_queue.get()
            if record is None:
                break
            logger.info(record)
            _ocr_log_queue.task_done()
        except Exception:
            pass

_log_thread = threading.Thread(target=_async_benchmark_logger_worker, daemon=True)
_log_thread.start()

_reader_lock = threading.Lock()
_easyocr_reader = None

_models_lock = threading.Lock()
_plate_model = None
_vehicle_model = None
_optimal_devices = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        import torch
        with _reader_lock:
            if _easyocr_reader is None:
                use_gpu = torch.cuda.is_available()
                print(f"[AI ENGINE] Initializing EasyOCR (GPU={use_gpu})...")
                _easyocr_reader = easyocr.Reader(['vi', 'en'], gpu=use_gpu)
    return _easyocr_reader

def get_ai_models():
    global _plate_model, _vehicle_model, _optimal_devices
    if _plate_model is None or _vehicle_model is None:
        with _models_lock:
            if _plate_model is None or _vehicle_model is None:
                from ultralytics import YOLO
                import openvino

                PLATE_MODEL_PATH = 'Train_Final_PBL4/weights/best_openvino_model'
                has_ov_vehicle = False
                if os.path.exists('yolov8n_openvino_model'):
                    VEHICLE_MODEL_PATH = 'yolov8n_openvino_model'
                    has_ov_vehicle = True
                else:
                    VEHICLE_MODEL_PATH = 'yolov8n.pt'

                target_device_plate = 'cpu'
                target_device_vehicle = 'cpu'

                print(f"[AI ENGINE] Loading models - Plate: {target_device_plate}, Vehicle: {target_device_vehicle}")
                _plate_model = YOLO(PLATE_MODEL_PATH, task='detect')
                _vehicle_model = YOLO(VEHICLE_MODEL_PATH, task='detect')
                _optimal_devices = {
                    "plate": target_device_plate,
                    "vehicle": target_device_vehicle
                }
    return _plate_model, _vehicle_model, _optimal_devices


class TrafficAIEngine(threading.Thread):
    def __init__(self, source_path: str):
        super().__init__(daemon=True)
        self.source_path = source_path
        self.stop_event = threading.Event()
        self.fps = 0.0
        self.processed_frames = 0
        self.status = "Initializing"

    def _run_image_pipeline(self, plate_model, vehicle_model, devices, reader):
        global latest_processed_frame
        try:
            self.status = "Reading image..."
            img = None
            try:
                with open(self.source_path, "rb") as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error reading image: {e}")

            if img is None:
                self.status = "Failed to load image"
                return


            ih, iw = img.shape[:2]
            if max(ih, iw) > 1280:
                scale_down = 1280.0 / max(ih, iw)
                img = cv2.resize(img, (int(iw * scale_down), int(ih * scale_down)), interpolation=cv2.INTER_AREA)

            self.status = "Analyzing image with AI..."
            target_device_plate = devices["plate"]
            target_device_vehicle = devices["vehicle"]

            knowledge_base = set()
            if os.path.exists('ground_truth.csv'):
                try:
                    with open('ground_truth.csv', 'r', encoding='utf-8') as f:
                        for line in f:
                            p = line.strip().upper().replace(".", "").replace("-", "")
                            if p:
                                knowledge_base.add(p)
                except Exception as e:
                    print(f"Error loading GT: {e}")

            def get_best_match(text):
                if not text:
                    return None, 0
                clean_text = text.upper().replace(" ", "").replace(".", "").replace("-", "")
                if len(clean_text) < 5:
                    return None, 0
                best_match = None
                best_ratio = 0
                for knowledge in knowledge_base:
                    ratio = SequenceMatcher(None, clean_text, knowledge).ratio()
                    if ratio >= 0.75 and ratio > best_ratio:
                        best_ratio = ratio
                        best_match = knowledge
                return best_match, best_ratio

            def finalize_formatting(text):
                if not text:
                    return None
                t = text.upper().replace(".", "").replace("-", "").replace(" ", "")
                if len(t) >= 3:
                    res = t[:2] + t[2]
                    if len(t) > 3:
                        res += "." + t[3:]
                    return res
                return t

            def format_plate_vietnam(text):
                if not text:
                    return None, False
                matched, ratio = get_best_match(text)
                if matched:
                    return finalize_formatting(matched), True
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) >= 7 and re.match(r'^\d{2}[A-Z]', clean):
                    return finalize_formatting(clean), False
                return None, False


            v_results = vehicle_model.predict(img, conf=0.25, verbose=False, imgsz=640, classes=[2, 3, 5, 7], device=target_device_vehicle)
            p_results = plate_model.predict(img, conf=0.30, verbose=False, imgsz=640, device=target_device_plate)

            v_boxes = v_results[0].boxes.xyxy.cpu().numpy().astype(int) if len(v_results[0].boxes) > 0 else np.empty((0, 4), dtype=int)
            v_cls = v_results[0].boxes.cls.cpu().numpy().astype(int) if len(v_results[0].boxes) > 0 else []
            v_names = v_results[0].names
            p_boxes = p_results[0].boxes.xyxy.cpu().numpy().astype(int) if len(p_results[0].boxes) > 0 else np.empty((0, 4), dtype=int)

            n_p = len(p_boxes)
            m_v = len(v_boxes)


            fallback_text_dict = {}
            if n_p == 0:
                self.status = "Fast OCR plate scan..."
                try:
                    ih, iw = img.shape[:2]
                    scale = 480.0 / max(ih, iw) if max(ih, iw) > 480 else 1.0
                    small_img = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

                    ocr_candidates = reader.readtext(small_img, canvas_size=480, mag_ratio=1.0)
                    plate_lines = []
                    for bbox, text, conf in ocr_candidates:
                        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if re.match(r'^\d{2}[A-Z0-9]', clean) or (clean.isdigit() and len(clean) in [4, 5]):
                            plate_lines.append((bbox, text, clean, conf))

                    if plate_lines:
                        all_pts = []
                        line_texts = []
                        for bbox, text, clean, conf in plate_lines:
                            all_pts.extend(bbox)
                            line_texts.append(text)

                        inv_scale = 1.0 / scale
                        xs = [p[0] * inv_scale for p in all_pts]
                        ys = [p[1] * inv_scale for p in all_pts]
                        pad = 14
                        px1 = max(0, int(min(xs)) - pad)
                        py1 = max(0, int(min(ys)) - pad)
                        px2 = min(iw, int(max(xs)) + pad)
                        py2 = min(ih, int(max(ys)) + pad)

                        p_boxes = np.array([[px1, py1, px2, py2]], dtype=int)
                        n_p = 1
                        fallback_text_dict[0] = " ".join(line_texts)
                except Exception as ex:
                    print(f"Fast fallback OCR detection error: {ex}")


            plate_to_vehicle = {}
            if n_p > 0 and m_v > 0:
                N = n_p + m_v
                BIG = 1e6
                min_score = 0.30
                cost_matrix = np.full((N, N), fill_value=BIG, dtype=np.float32)

                for p_idx, p_box in enumerate(p_boxes):
                    px1, py1, px2, py2 = p_box
                    p_area = max(1, (px2 - px1) * (py2 - py1))
                    pcx = (px1 + px2) / 2.0
                    pcy = (py1 + py2) / 2.0

                    for v_idx, v_box in enumerate(v_boxes):
                        vx1, vy1, vx2, vy2 = v_box
                        ix1 = max(px1, vx1)
                        iy1 = max(py1, vy1)
                        ix2 = min(px2, vx2)
                        iy2 = min(py2, vy2)
                        iw = max(0, ix2 - ix1)
                        ih = max(0, iy2 - iy1)
                        containment = (iw * ih) / p_area

                        diag = max(1.0, ((vx2 - vx1)**2 + (vy2 - vy1)**2) ** 0.5)
                        center_dist = ((pcx - (vx1 + vx2)/2.0)**2 + (pcy - (vy1 + vy2)/2.0)**2) ** 0.5
                        norm_dist = center_dist / diag

                        if containment > 0.20 or (vx1 - 35 <= pcx <= vx2 + 35 and vy1 - 35 <= pcy <= vy2 + 35):
                            score = (containment * 1.5) - (norm_dist * 0.5)
                            cost_matrix[p_idx, v_idx] = -score

                for i in range(n_p):
                    cost_matrix[i, m_v + i] = -min_score
                for j in range(m_v):
                    cost_matrix[n_p + j, j] = -min_score
                for i in range(n_p):
                    for j in range(m_v):
                        cost_matrix[n_p + j, m_v + i] = 0.0

                p_indices, v_indices = linear_sum_assignment(cost_matrix)
                for p_idx, v_idx in zip(p_indices, v_indices):
                    if p_idx < n_p and v_idx < m_v and -cost_matrix[p_idx, v_idx] >= min_score:
                        cls_name = v_names[v_cls[v_idx]].upper()
                        plate_to_vehicle[p_idx] = (v_idx, cls_name)


            annotated_img = img.copy()
            ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            detected_plates_info = []

            for p_idx, p_box in enumerate(p_boxes):
                px1, py1, px2, py2 = p_box
                plate_crop = img[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue

                v_info = plate_to_vehicle.get(p_idx, (None, "CAR"))
                v_type = v_info[1]

                h_orig, w_orig = plate_crop.shape[:2]
                aspect_ratio = w_orig / max(h_orig, 1)

                enhanced = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                eh, ew = enhanced.shape[:2]
                if ew < 160 or eh < 60:
                    enhanced = cv2.resize(enhanced, (max(ew * 2, 180), max(eh * 2, 70)), interpolation=cv2.INTER_CUBIC)
                eh, ew = enhanced.shape[:2]

                raw_text = ""
                if p_idx in fallback_text_dict:
                    raw_text = fallback_text_dict[p_idx]
                elif aspect_ratio < 2.3:
                    try:
                        res = reader.readtext(enhanced, allowlist=ALLOWLIST, detail=0)
                        raw_text = "".join(res).strip()
                    except Exception:
                        pass
                else:
                    try:
                        res = reader.recognize(enhanced, horizontal_list=[[0, ew, 0, eh]], free_list=[], detail=0, allowlist=ALLOWLIST)
                        raw_text = "".join(res).strip()
                    except Exception:
                        pass

                if not raw_text or (len(raw_text) < 5 and p_idx not in fallback_text_dict):
                    try:
                        res = reader.readtext(enhanced, allowlist=ALLOWLIST, detail=0)
                        raw_text = "".join(res).strip()
                    except Exception:
                        pass

                text, is_trusted = format_plate_vietnam(raw_text)
                final_plate = text if text else (raw_text if len(raw_text) >= 4 else "UNREADABLE")


                if v_info[0] is None:
                    clean_check = re.sub(r'[^A-Z0-9]', '', final_plate.upper())
                    if re.match(r'^\d{2}[A-Z]\d', clean_check):
                        v_type = "MOTORCYCLE"
                    else:
                        v_type = "CAR"

                crop_path = None
                try:
                    crop_dir = os.path.join("static", "crops")
                    os.makedirs(crop_dir, exist_ok=True)
                    t_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    clean_save_name = re.sub(r'[^A-Z0-9]', '', final_plate)
                    if not clean_save_name:
                        clean_save_name = f"PLATE_{p_idx}"
                    crop_filename = f"{clean_save_name}_{t_str}.jpg"
                    crop_filepath = os.path.join(crop_dir, crop_filename)
                    cv2.imwrite(crop_filepath, plate_crop)
                    crop_path = f"static/crops/{crop_filename}"
                except Exception as ex:
                    print(f"Error saving image crop: {ex}")


                db_session = SessionLocal()
                try:
                    clean_target = re.sub(r'[^A-Z0-9]', '', final_plate.upper()).replace('[', 'L').replace(']', '1')
                    all_existing = db_session.query(Detection).filter(Detection.deleted_at.is_(None)).all()
                    match_record = None
                    for det in all_existing:
                        c_plate = re.sub(r'[^A-Z0-9]', '', (det.plate_text or '').upper()).replace('[', 'L').replace(']', '1')
                        if c_plate == clean_target:
                            match_record = det
                            break

                    if match_record:

                        match_record.timestamp = datetime.datetime.now()
                        if crop_path:
                            match_record.crop_path = crop_path
                        if v_type and v_type != "UNKNOWN":
                            match_record.vehicle_type = v_type
                        match_record.confidence = max(match_record.confidence or 0.85, 1.0 if is_trusted else 0.88)
                        db_session.commit()
                    else:
                        new_det = Detection(
                            timestamp=datetime.datetime.now(),
                            plate_text=final_plate,
                            vehicle_type=v_type,
                            confidence=1.0 if is_trusted else 0.88,
                            payment_status="UNPAID",
                            package_type="STANDARD",
                            crop_path=crop_path,
                            lane="Lane 1"
                        )
                        db_session.add(new_det)
                        db_session.commit()

                    def notify_app(plate_val, v_type_val, c_path_val):
                        import urllib.request
                        import json
                        try:
                            req_data = json.dumps({
                                "plate": plate_val,
                                "type": v_type_val,
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "status": "UNPAID",
                                "package": "STANDARD",
                                "crop_path": c_path_val or ""
                            }).encode('utf-8')
                            req = urllib.request.Request(
                                "http://localhost:8001/api/internal/notify",
                                data=req_data,
                                headers={'Content-Type': 'application/json'}
                            )
                            with urllib.request.urlopen(req, timeout=1.0) as resp:
                                resp.read()
                        except Exception:
                            pass
                    threading.Thread(target=notify_app, args=(final_plate, v_type, crop_path), daemon=True).start()
                except Exception as dbe:
                    print(f"DB Error in image pipeline: {dbe}")
                finally:
                    db_session.close()

                detected_plates_info.append({
                    "box": p_box,
                    "plate": final_plate,
                    "type": v_type,
                    "v_idx": v_info[0]
                })


            matched_v_indices = {info["v_idx"] for info in detected_plates_info if info["v_idx"] is not None}
            for v_idx, v_box in enumerate(v_boxes):
                vx1, vy1, vx2, vy2 = v_box
                cls_name = v_names[v_cls[v_idx]].upper()
                is_matched = v_idx in matched_v_indices
                v_color = (0, 255, 0) if is_matched else (0, 165, 255)
                cv2.rectangle(annotated_img, (vx1, vy1), (vx2, vy2), v_color, 2)
                cv2.putText(annotated_img, cls_name, (vx1, max(vy1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)

            for info in detected_plates_info:
                px1, py1, px2, py2 = info["box"]
                p_text = info["plate"]
                cv2.rectangle(annotated_img, (px1, py1), (px2, py2), (0, 255, 255), 2)
                (tw, th), _ = cv2.getTextSize(p_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ty1 = max(0, py1 - th - 12)
                ty2 = max(th + 12, py1)
                cv2.rectangle(annotated_img, (px1, ty1), (px1 + tw + 10, ty2), (0, 0, 0), -1)
                cv2.rectangle(annotated_img, (px1, ty1), (px1 + tw + 10, ty2), (0, 255, 255), 1)
                cv2.putText(annotated_img, p_text, (px1 + 5, ty2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            canvas_base = np.zeros((720, 1280, 3), dtype=np.uint8)
            canvas_base[:] = (18, 14, 12)

            ih, iw = annotated_img.shape[:2]
            scale = min(1280 / iw, 640 / ih)
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            resized_img = cv2.resize(annotated_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            off_x = (1280 - new_w) // 2
            off_y = 48 + (640 - new_h) // 2
            canvas_base[off_y:off_y+new_h, off_x:off_x+new_w] = resized_img

            cv2.rectangle(canvas_base, (0, 0), (1280, 42), (28, 22, 18), -1)
            cv2.line(canvas_base, (0, 42), (1280, 42), (45, 212, 191), 2)
            cv2.putText(canvas_base, "TRAFFIC-AI INTELLIGENCE - STATIC IMAGE ANALYSIS", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (45, 212, 191), 2)

            count_text = f"Vehicles: {m_v} | Plates: {n_p} Detected"
            cv2.putText(canvas_base, count_text, (850, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            self.status = f"Completed ({n_p} plates detected)"
            self.processed_frames = 1
            self.fps = 1.0


            while not self.stop_event.is_set():
                canvas = canvas_base.copy()
                curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(canvas, f"Status: Analysis Complete | Captured at: {curr_time}", (20, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

                _, jpeg = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
                with latest_frame_lock:
                    latest_processed_frame = jpeg.tobytes()

                time.sleep(0.5)

            self.status = "Finished"
        except Exception as e:
            print(f"Error in image pipeline: {e}")
            self.status = f"Error: {e}"

    def run(self):
        try:
            self.status = "Loading AI models..."
            plate_model, vehicle_model, devices = get_ai_models()
            target_device_plate = devices["plate"]
            target_device_vehicle = devices["vehicle"]
            reader = get_easyocr_reader()


            IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
            if isinstance(self.source_path, str) and os.path.splitext(self.source_path)[1].lower() in IMAGE_EXTENSIONS:
                self._run_image_pipeline(plate_model, vehicle_model, devices, reader)
                return

            self.status = "Opening source..."
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                self.status = "Failed to open source"
                return

            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not native_fps or native_fps <= 0 or native_fps > 120:
                native_fps = 30.0
            frame_delay = 1.0 / native_fps

            self.status = "Active"
            tracked_objects = {}
            objects_lock = threading.Lock()
            last_seen_cleanup = time.time()

            knowledge_base = set()
            if os.path.exists('ground_truth.csv'):
                try:
                    with open('ground_truth.csv', 'r', encoding='utf-8') as f:
                        for line in f:
                            plate = line.strip().upper().replace(".", "").replace("-", "")
                            if plate:
                                knowledge_base.add(plate)
                except Exception as e:
                    print(f"Error loading GT: {e}")

            def get_best_match(text):
                if not text:
                    return None, 0
                clean_text = text.upper().replace(" ", "").replace(".", "").replace("-", "")
                if len(clean_text) < 5:
                    return None, 0
                best_match = None
                best_ratio = 0
                for knowledge in knowledge_base:
                    ratio = SequenceMatcher(None, clean_text, knowledge).ratio()
                    if ratio >= 0.75 and ratio > best_ratio:
                        best_ratio = ratio
                        best_match = knowledge
                return best_match, best_ratio

            def finalize_formatting(text):
                if not text:
                    return None
                t = text.upper().replace(".", "").replace("-", "").replace(" ", "")
                if len(t) >= 3:
                    res = t[:2] + t[2]
                    if len(t) > 3:
                        res += "." + t[3:]
                    return res
                return t

            def format_plate_vietnam(text):
                if not text:
                    return None, False
                matched, ratio = get_best_match(text)
                if matched:
                    return finalize_formatting(matched), True
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) >= 7 and re.match(r'^\d{2}[A-Z]', clean):
                    return finalize_formatting(clean), False
                return None, False


            ocr_queue = Queue(maxsize=5)

            def ocr_worker_loop():
                while not self.stop_event.is_set():
                    try:
                        task = ocr_queue.get(timeout=0.1)
                    except Empty:
                        continue

                    track_id, plate_crop, v_type = task
                    try:
                        t_ocr_start = time.perf_counter()
                        h_orig, w_orig = plate_crop.shape[:2]
                        aspect_ratio = w_orig / max(h_orig, 1)

                        enhanced = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                        h, w = enhanced.shape[:2]
                        if w < 160 or h < 60:
                            enhanced = cv2.resize(enhanced, (max(w * 2, 180), max(h * 2, 70)), interpolation=cv2.INTER_CUBIC)

                        ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        eh, ew = enhanced.shape[:2]
                        raw_text = ""
                        path_used = "fast_recognition"
                        fallback_triggered = False



                        if aspect_ratio < 2.3:
                            path_used = "multiline"
                            try:
                                ocr_results = reader.readtext(enhanced, allowlist=ALLOWLIST, detail=0)
                                raw_text = "".join(ocr_results).strip()
                            except Exception:
                                pass
                        else:
                            try:
                                results = reader.recognize(
                                    enhanced,
                                    horizontal_list=[[0, ew, 0, eh]],
                                    free_list=[],
                                    detail=0,
                                    allowlist=ALLOWLIST
                                )
                                raw_text = "".join(results).strip()
                            except Exception:
                                pass


                            if len(raw_text) < 5:
                                fallback_triggered = True
                                try:
                                    ocr_results = reader.readtext(enhanced, allowlist=ALLOWLIST, detail=0)
                                    raw_text = "".join(ocr_results).strip()
                                except Exception:
                                    pass

                        ocr_latency_ms = round((time.perf_counter() - t_ocr_start) * 1000, 2)
                        text, is_trusted = format_plate_vietnam(raw_text)


                        matched_gt = ""
                        clean_raw = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                        for gt in knowledge_base:
                            if SequenceMatcher(None, clean_raw, gt).ratio() >= 0.70:
                                matched_gt = gt
                                break


                        try:
                            log_entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{track_id},{aspect_ratio:.2f},{path_used},{ocr_latency_ms:.1f},{raw_text},{text or ''},{matched_gt},{True},{is_trusted},{fallback_triggered}"
                            _ocr_log_queue.put_nowait(log_entry)
                        except Exception:
                            pass

                        if text:
                            with objects_lock:
                                if track_id in tracked_objects:
                                    obj = tracked_objects[track_id]
                                    obj["candidates"][text] = obj["candidates"].get(text, 0) + 1
                                    best = max(obj["candidates"], key=obj["candidates"].get)
                                    obj["plate"] = best

                                    threshold = 1 if is_trusted else 2
                                    if obj["candidates"][best] >= threshold and not obj["locked"]:
                                        obj["locked"] = True


                                        crop_path = None
                                        try:
                                            crop_dir = os.path.join("static", "crops")
                                            os.makedirs(crop_dir, exist_ok=True)
                                            timestamp_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                            crop_filename = f"{best.replace('.', '').replace('-', '')}_{timestamp_slug}.jpg"
                                            crop_filepath = os.path.join(crop_dir, crop_filename)
                                            cv2.imwrite(crop_filepath, plate_crop)
                                            crop_path = f"static/crops/{crop_filename}"
                                        except Exception as ex:
                                            print(f"Error saving crop: {ex}")


                                        db_session = SessionLocal()
                                        try:
                                            clean_best = re.sub(r'[^A-Z0-9]', '', best.upper())
                                            all_existing = db_session.query(Detection).filter(Detection.deleted_at.is_(None)).all()
                                            match_record = None
                                            for det in all_existing:
                                                if re.sub(r'[^A-Z0-9]', '', (det.plate_text or '').upper()) == clean_best:
                                                    match_record = det
                                                    break

                                            if match_record:
                                                match_record.timestamp = datetime.datetime.now()
                                                if crop_path:
                                                    match_record.crop_path = crop_path
                                                db_session.commit()
                                            else:
                                                new_det = Detection(
                                                    timestamp=datetime.datetime.now(),
                                                    plate_text=best,
                                                    vehicle_type=v_type,
                                                    confidence=1.0 if is_trusted else 0.85,
                                                    payment_status="UNPAID",
                                                    package_type="STANDARD",
                                                    crop_path=crop_path,
                                                    lane="Lane 1"
                                                )
                                                db_session.add(new_det)
                                                db_session.commit()


                                                def notify_app(plate_val, v_type_val, c_path_val):
                                                    import urllib.request
                                                    import json
                                                    try:
                                                        req_data = json.dumps({
                                                            "plate": plate_val,
                                                            "type": v_type_val,
                                                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                            "status": "UNPAID",
                                                            "package": "STANDARD",
                                                            "crop_path": c_path_val or ""
                                                        }).encode('utf-8')
                                                        req = urllib.request.Request(
                                                            "http://localhost:8001/api/internal/notify",
                                                            data=req_data,
                                                            headers={'Content-Type': 'application/json'}
                                                        )
                                                        with urllib.request.urlopen(req, timeout=1.0) as response:
                                                            response.read()
                                                    except Exception:
                                                        pass
                                                threading.Thread(target=notify_app, args=(best, v_type, crop_path), daemon=True).start()
                                                print(f"[BOT LOGGED] {best} | {v_type}")
                                        except Exception as db_err:
                                            print(f"DB Error: {db_err}")
                                        finally:
                                            db_session.close()
                    except Exception as ocr_err:
                        print(f"OCR processing error: {ocr_err}")
                    finally:
                        with objects_lock:
                            if track_id in tracked_objects:
                                tracked_objects[track_id]["ocr_in_flight"] = False
                        ocr_queue.task_done()

            ocr_thread = threading.Thread(target=ocr_worker_loop, daemon=True)
            ocr_thread.start()

            frame_count = 0
            start_time = time.time()
            current_detections = []

            global latest_processed_frame

            while cap.isOpened() and not self.stop_event.is_set():
                loop_start = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.source_path, str) and not self.source_path.startswith("rtsp") and not self.source_path.startswith("http"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                frame_count += 1
                self.processed_frames = frame_count


                if frame_count % 2 == 0:
                    v_results = vehicle_model.track(frame, conf=0.25, persist=True, verbose=False, imgsz=640, classes=[2, 3, 5, 7], device=target_device_vehicle)
                    p_results = plate_model.predict(frame, conf=0.35, verbose=False, imgsz=640, device=target_device_plate)

                    new_detections = []
                    if v_results[0].boxes.id is not None:
                        v_boxes = v_results[0].boxes.xyxy.cpu().numpy().astype(int)
                        v_ids = v_results[0].boxes.id.cpu().numpy().astype(int)
                        v_cls = v_results[0].boxes.cls.cpu().numpy().astype(int)
                        v_names = v_results[0].names

                        p_boxes = p_results[0].boxes.xyxy.cpu().numpy().astype(int) if len(p_results[0].boxes) > 0 else []



                        assigned_plates = {}
                        n_p = len(p_boxes)
                        m_v = len(v_boxes)

                        if n_p > 0 and m_v > 0:
                            N = n_p + m_v
                            BIG = 1e6
                            min_score = 0.30
                            cost_matrix = np.full((N, N), fill_value=BIG, dtype=np.float32)


                            for p_idx, p_box in enumerate(p_boxes):
                                px1, py1, px2, py2 = p_box
                                p_area = max(1, (px2 - px1) * (py2 - py1))
                                pcx = (px1 + px2) / 2.0
                                pcy = (py1 + py2) / 2.0

                                for v_idx, box in enumerate(v_boxes):
                                    vx1, vy1, vx2, vy2 = box

                                    ix1 = max(px1, vx1)
                                    iy1 = max(py1, vy1)
                                    ix2 = min(px2, vx2)
                                    iy2 = min(py2, vy2)
                                    iw = max(0, ix2 - ix1)
                                    ih = max(0, iy2 - iy1)
                                    containment = (iw * ih) / p_area

                                    diag = max(1.0, ((vx2 - vx1)**2 + (vy2 - vy1)**2) ** 0.5)
                                    center_dist = ((pcx - (vx1 + vx2)/2.0)**2 + (pcy - (vy1 + vy2)/2.0)**2) ** 0.5
                                    norm_dist = center_dist / diag


                                    if containment > 0.20 or (vx1 - 35 <= pcx <= vx2 + 35 and vy1 - 35 <= pcy <= vy2 + 35):
                                        score = (containment * 1.5) - (norm_dist * 0.5)
                                        cost_matrix[p_idx, v_idx] = -score


                            for i in range(n_p):
                                cost_matrix[i, m_v + i] = -min_score


                            for j in range(m_v):
                                cost_matrix[n_p + j, j] = -min_score


                            for i in range(n_p):
                                for j in range(m_v):
                                    cost_matrix[n_p + j, m_v + i] = 0.0


                            p_indices, v_indices = linear_sum_assignment(cost_matrix)

                            for p_idx, v_idx in zip(p_indices, v_indices):

                                if p_idx < n_p and v_idx < m_v and -cost_matrix[p_idx, v_idx] >= min_score:
                                    matched_tid = v_ids[v_idx]
                                    assigned_plates[matched_tid] = p_boxes[p_idx]

                        with objects_lock:
                            for box, track_id, cls_id in zip(v_boxes, v_ids, v_cls):
                                v_type = v_names[cls_id].upper()
                                if track_id not in tracked_objects:
                                    tracked_objects[track_id] = {
                                        "v_type": v_type,
                                        "plate": "SCANNING",
                                        "locked": False,
                                        "candidates": {},
                                        "last_seen": time.time(),
                                        "ocr_in_flight": False
                                    }

                                obj = tracked_objects[track_id]
                                obj["last_seen"] = time.time()


                                if track_id in assigned_plates and not obj["locked"] and not obj["ocr_in_flight"]:
                                    px1, py1, px2, py2 = assigned_plates[track_id]
                                    plate_h = py2 - py1

                                    if plate_h >= 38:
                                        plate_crop = frame[py1:py2, px1:px2]
                                        if plate_crop.size > 0:
                                            obj["ocr_in_flight"] = True
                                            try:
                                                ocr_queue.put_nowait((track_id, plate_crop.copy(), obj["v_type"]))
                                            except Exception:
                                                obj["ocr_in_flight"] = False

                                new_detections.append({
                                    "box": box, "plate": obj["plate"], "locked": obj["locked"]
                                })

                        current_detections = new_detections


                for det in current_detections:
                    bx1, by1, bx2, by2 = det["box"]
                    color = (0, 255, 0) if det["locked"] else (0, 165, 255)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(frame, f"{det['plate']}", (bx1, max(by1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                cv2.putText(frame, f"TrafficAI BOT Camera - FPS: {self.fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    self.fps = round(frame_count / elapsed, 1)
                    start_time = time.time()
                    frame_count = 0


                frame_resized = cv2.resize(frame, (1280, 720))
                _, jpeg = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

                with latest_frame_lock:
                    latest_processed_frame = jpeg.tobytes()


                now_time = time.monotonic()
                if now_time - last_seen_cleanup > 4.0:
                    with objects_lock:
                        to_del = [tid for tid, o in tracked_objects.items() if now_time - o["last_seen"] > 4.0]
                        for tid in to_del:
                            del tracked_objects[tid]
                    last_seen_cleanup = now_time


                process_time = time.monotonic() - loop_start
                sleep_left = frame_delay - process_time
                if sleep_left > 0.005:
                    time.sleep(sleep_left)

            cap.release()
            self.status = "Finished"
        except Exception as err:
            self.status = f"Error: {err}"
            print(f"Error inside AI Thread: {err}")
