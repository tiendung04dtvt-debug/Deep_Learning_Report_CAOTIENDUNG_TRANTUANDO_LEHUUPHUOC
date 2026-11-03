"""TrafficAI dashboard backend."""

import csv
import io
import os
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, List

import jwt
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

from config import settings
from src.database import engine, SessionLocal, get_db, Base
from src.models import (
    User,
    UserSession,
    Detection,
    DetectionHistory,
    SystemConfig,
    Alert,
    AuditLog
)
from src.shield import TrafficShield
from src.engine import (
    TrafficAIEngine,
    get_ai_models,
    get_easyocr_reader,
    latest_frame_lock
)
import src.engine as engine_module

VALID_STATUSES = {"UNPAID", "PAID", "BLACKLISTED"}
CONFIG_KEYS = {"system_name", "price_per_vehicle", "token_expire_h"}

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class PackageUpdate(BaseModel):
    package: str = Field("SUPREME", min_length=1, max_length=32)


class BlacklistUpdate(BaseModel):
    reason: str = Field("Manual review", max_length=255)


class ConfigUpdate(BaseModel):
    value: Any


pwd_context = CryptContext(schemes=["argon2", "pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")
scheduler = BackgroundScheduler()
traffic_shield = TrafficShield()


def get_client_ip_from_request(request: Request) -> str:
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip() or "unknown"
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip() or "unknown"
    return request.client.host if request.client else "unknown"


def get_client_ip_from_websocket(websocket: WebSocket) -> str:
    if settings.TRUST_PROXY_HEADERS:
        forwarded = websocket.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip() or "unknown"
        real_ip = websocket.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip() or "unknown"
    return websocket.client.host if websocket.client else "unknown"


def parse_dt(value):
    if value is None or isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(value).replace("Z", ""), fmt)
        except ValueError:
            continue
    return None


def iso_dt(value):
    parsed = parse_dt(value)
    return parsed.isoformat() if parsed else ""


def get_config_value(db: Session, key: str, default: str = "") -> str:
    cfg = db.query(SystemConfig).filter(SystemConfig.key == key).first()
    return cfg.value if cfg else default


def normalize_plate(plate: str) -> str:
    value = (plate or "").strip().upper()
    if not value:
        raise HTTPException(status_code=400, detail="Plate is required")
    return value


def normalize_status(status: str) -> str:
    value = (status or "").strip().upper()
    if value and value not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Unsupported status: {value}")
    return value


def normalize_package(package: str) -> str:
    value = (package or "STANDARD").strip().upper()
    if not re.fullmatch(r"[A-Z0-9_-]{1,32}", value):
        raise HTTPException(status_code=400, detail="Package must contain only letters, numbers, '-' or '_'")
    return value


def safe_int(value: Any, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    if max_value is not None and parsed > max_value:
        return default
    return parsed


def get_price_per_vehicle(db: Session) -> int:
    return safe_int(get_config_value(db, "price_per_vehicle", settings.PRICE_PER_VEHICLE), settings.PRICE_PER_VEHICLE, 0)


def get_token_expire_hours(db: Session) -> int:
    return safe_int(get_config_value(db, "token_expire_h", settings.TOKEN_EXPIRE_H), settings.TOKEN_EXPIRE_H, 1, 168)


def detection_payload(row: Detection) -> dict:
    return {
        "id": row.id,
        "plate": row.plate_text,
        "type": row.vehicle_type,
        "timestamp": iso_dt(row.timestamp),
        "status": row.payment_status,
        "package": row.package_type,
        "confidence": row.confidence,
        "speed_kmh": row.speed_kmh,
        "lane": row.lane,
        "blacklist_reason": row.blacklist_reason,
        "deleted_at": iso_dt(row.deleted_at),
        "crop_path": row.crop_path or "",
    }


def log_action(db: Session, user: str, action: str, detail: str = "", ip: str = ""):
    db.add(AuditLog(username=user, action=action, detail=detail, ip=ip))


def migrate_legacy_schema():
    conn = sqlite3.connect("traffic_monitoring.db")
    cur = conn.cursor()
    migrations = [
        "ALTER TABLE detections ADD COLUMN payment_status TEXT DEFAULT 'UNPAID'",
        "ALTER TABLE detections ADD COLUMN package_type TEXT DEFAULT 'STANDARD'",
        "ALTER TABLE detections ADD COLUMN blacklist_reason TEXT",
        "ALTER TABLE detections ADD COLUMN lane TEXT",
        "ALTER TABLE detections ADD COLUMN speed_kmh REAL",
        "ALTER TABLE detections ADD COLUMN deleted_at DATETIME",
        "ALTER TABLE detections ADD COLUMN crop_path TEXT",
        "ALTER TABLE detections ADD COLUMN direction TEXT",
    ]
    for sql in migrations:
        try:
            cur.execute(sql)
        except sqlite3.OperationalError:
            pass
    try:
        cur.execute("UPDATE detections SET payment_status = 'UNPAID' WHERE payment_status IS NULL OR payment_status = ''")
        cur.execute("UPDATE detections SET package_type = 'STANDARD' WHERE package_type IS NULL OR package_type = '' OR package_type = 'NONE'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    migrate_legacy_schema()

    db = SessionLocal()
    defaults = {
        "price_per_vehicle": str(settings.PRICE_PER_VEHICLE),
        "token_expire_h": str(settings.TOKEN_EXPIRE_H),
        "system_name": settings.SYSTEM_NAME,
    }
    for key, value in defaults.items():
        if not db.query(SystemConfig).filter(SystemConfig.key == key).first():
            db.add(SystemConfig(key=key, value=value, updated_by="SYSTEM"))
    if not db.query(User).filter(User.username == "admin").first():
        db.add(User(username="admin", hashed_password=pwd_context.hash("123456"), role="admin"))
    db.commit()
    db.close()

    if not scheduler.running:
        scheduler.start()

    # Pre-load AI models and reader in the background during server boot so start is instant!
    def pre_load():
        try:
            p_m, v_m, _ = get_ai_models()
            rdr = get_easyocr_reader()
            # Perform 1-shot full neural network warmup on CPU so inference is truly instant (<500ms)
            import numpy as np
            import cv2
            dummy_f = np.zeros((640, 640, 3), dtype=np.uint8)
            p_m.predict(dummy_f, device='cpu', verbose=False)
            v_m.predict(dummy_f, device='cpu', verbose=False)
            dummy_p = np.zeros((80, 180, 3), dtype=np.uint8)
            rdr.readtext(dummy_p, canvas_size=480, mag_ratio=1.0)
            print("[AI ENGINE] All 3 AI models pre-loaded and fully warmed up on CPU in background!")
        except Exception as e:
            print(f"[AI ENGINE] Error pre-loading models in background: {e}")
    threading.Thread(target=pre_load, daemon=True).start()

    yield
    if scheduler.running:
        scheduler.shutdown()


app = FastAPI(title=settings.SYSTEM_NAME, version=settings.VERSION, lifespan=lifespan)


@app.middleware("http")
async def request_logging_and_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    client_ip = get_client_ip_from_request(request)
    allowed, status_code, detail, retry_after = traffic_shield.check_http(
        client_ip,
        request.url.path,
        request.method,
        request.headers.get("content-length"),
    )
    if not allowed:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail, "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
        if retry_after:
            response.headers["Retry-After"] = str(retry_after)
        response.headers["X-TrafficShield"] = "blocked"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        print(f"ID: {request_id} | {request.method} {request.url.path} | IP: {client_ip} | Status: {status_code} | BLOCKED")
        return response

    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    print(f"ID: {request_id} | {request.method} {request.url.path} | IP: {client_ip} | Status: {response.status_code} | {process_time:.2f}ms")
    response.headers["X-Request-ID"] = request_id
    response.headers["X-TrafficShield"] = "pass"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_access_token(data: dict, expire_hours: int):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(hours=expire_hours)})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def require_role(roles: List[str]):
    def role_checker(user: dict = Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Permission denied")
        return user
    return role_checker


@app.get("/api/health")
def health_check():
    # Return 503 until background AI model pre-loading has completed
    if engine_module._plate_model is None or engine_module._vehicle_model is None or engine_module._easyocr_reader is None:
        raise HTTPException(status_code=503, detail="AI models initializing...")
    return {"status": "healthy", "version": settings.VERSION, "engine": "TrafficAI", "models_ready": True}


@app.get("/api/system/status")
def get_system_status(db: Session = Depends(get_db), _: dict = Depends(require_role(["admin"]))):
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    return {
        "version": settings.VERSION,
        "database": db_path,
        "database_size_bytes": db_size,
        "detections": db.query(Detection).count(),
        "active_sessions": db.query(UserSession).filter(UserSession.is_active).count(),
        "scheduler_running": scheduler.running,
        "allowed_origins": settings.ALLOWED_ORIGINS,
        "traffic_shield": traffic_shield.snapshot(),
    }


@app.get("/api/security/status")
def get_security_status(_: dict = Depends(require_role(["admin"]))):
    return traffic_shield.snapshot()


@app.post("/api/login")
async def login(request: Request, data: LoginRequest, db: Session = Depends(get_db)):
    username = data.username.strip()
    password = data.password
    client_ip = get_client_ip_from_request(request)
    user = db.query(User).filter(User.username == username, User.is_active).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        log_action(db, username or "unknown", "LOGIN_FAIL", ip=client_ip)
        db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username, "role": user.role, "uid": user.id}, get_token_expire_hours(db))
    db.add(UserSession(user_id=user.id, ip=client_ip, user_agent=request.headers.get("user-agent")))
    log_action(db, user.username, "LOGIN_OK", ip=client_ip)
    db.commit()
    return {"token": token, "username": user.username, "role": user.role}


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    live = Detection.deleted_at.is_(None)
    all_rows = db.query(Detection).filter(live).order_by(desc(Detection.timestamp), desc(Detection.id)).all()
    
    # Deduplicate strictly across unique license plates
    unique_rows = []
    seen_plates = set()
    cars = 0
    paid = 0
    blacklisted = 0
    for row in all_rows:
        clean = re.sub(r'[^A-Z0-9]', '', (row.plate_text or '').upper()).replace('[', 'L').replace(']', '1')
        if clean and clean not in seen_plates:
            seen_plates.add(clean)
            unique_rows.append(row)
            if (row.vehicle_type or "").upper() == "CAR":
                cars += 1
            if row.payment_status == "PAID":
                paid += 1
            elif row.payment_status == "BLACKLISTED":
                blacklisted += 1

    total = len(unique_rows)
    motorcycles = max(0, total - cars)
    unpaid = max(0, total - paid - blacklisted)
    price = get_price_per_vehicle(db)
    revenue_raw = paid * price
    return {
        "total": total,
        "cars": cars,
        "motorcycles": motorcycles,
        "confidence": "98.5%",
        "paid": paid,
        "blacklisted": blacklisted,
        "unpaid": unpaid,
        "revenue": f"{revenue_raw:,} VND",
        "revenue_raw": revenue_raw,
    }


@app.get("/api/vehicles")
def get_vehicles(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    search: str = "",
    status: str = "",
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_user),
):
    query = db.query(Detection).filter(Detection.deleted_at.is_(None))
    if search:
        query = query.filter(Detection.plate_text.contains(search.strip().upper()))

    all_rows = query.order_by(desc(Detection.timestamp), desc(Detection.id)).all()
    
    # Deduplicate strictly: keep only 1 newest record per unique license plate
    unique_rows = []
    seen_plates = set()
    for row in all_rows:
        clean = re.sub(r'[^A-Z0-9]', '', (row.plate_text or '').upper()).replace('[', 'L').replace(']', '1')
        if clean and clean not in seen_plates:
            seen_plates.add(clean)
            unique_rows.append(row)
            
    total = len(unique_rows)
    start_idx = (page - 1) * limit
    page_rows = unique_rows[start_idx:start_idx + limit]
    
    return {
        "data": [detection_payload(row) for row in page_rows],
        "total": total,
        "page": page,
        "pages": max(1, (total + limit - 1) // limit),
    }


@app.get("/api/vehicle/{plate}/history")
def get_vehicle_history(plate: str, db: Session = Depends(get_db), _: dict = Depends(get_current_user)):
    plate = normalize_plate(plate)
    detections = (
        db.query(Detection)
        .filter(Detection.plate_text == plate)
        .order_by(desc(Detection.timestamp), desc(Detection.id))
        .limit(50)
        .all()
    )
    changes = (
        db.query(DetectionHistory)
        .filter(DetectionHistory.plate_text == plate)
        .order_by(desc(DetectionHistory.changed_at), desc(DetectionHistory.id))
        .limit(50)
        .all()
    )
    if not detections and not changes:
        raise HTTPException(status_code=404, detail="Vehicle history not found")
    return {
        "plate": plate,
        "detections": [detection_payload(row) for row in detections],
        "changes": [
            {
                "old_status": row.old_status,
                "new_status": row.new_status,
                "changed_by": row.changed_by,
                "changed_at": iso_dt(row.changed_at),
            }
            for row in changes
        ],
    }


@app.get("/api/export/vehicles")
def export_vehicles(
    search: str = "",
    status: str = "",
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_user),
):
    status = normalize_status(status)
    query = db.query(Detection).filter(Detection.deleted_at.is_(None))
    if search:
        query = query.filter(Detection.plate_text.contains(search.strip().upper()))
    if status:
        query = query.filter(Detection.payment_status == status)
    rows = query.order_by(desc(Detection.timestamp), desc(Detection.id)).limit(10000).all()

    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "plate", "vehicle_type", "status", "package", "confidence", "speed_kmh", "lane", "blacklist_reason"])
    for row in rows:
        writer.writerow([
            row.id,
            iso_dt(row.timestamp),
            row.plate_text,
            row.vehicle_type,
            row.payment_status,
            row.package_type,
            row.confidence,
            row.speed_kmh,
            row.lane,
            row.blacklist_reason,
        ])

    filename = f"trafficai_vehicles_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/reports/overview")
def get_report_overview(
    days: int = Query(14, ge=1, le=90),
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_user),
):
    price = get_price_per_vehicle(db)
    start_date = (datetime.utcnow() - timedelta(days=days - 1)).date()
    daily = {
        (start_date + timedelta(days=offset)).isoformat(): {
            "date": (start_date + timedelta(days=offset)).isoformat(),
            "total": 0,
            "paid": 0,
            "unpaid": 0,
            "blacklisted": 0,
            "revenue": 0,
        }
        for offset in range(days)
    }
    status_counts = {status: 0 for status in VALID_STATUSES}
    vehicle_types: dict[str, int] = {}

    rows = db.query(Detection).filter(Detection.deleted_at.is_(None)).all()
    for row in rows:
        status = row.payment_status if row.payment_status in VALID_STATUSES else "UNPAID"
        status_counts[status] += 1
        vehicle_type = row.vehicle_type or "UNKNOWN"
        vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1

        timestamp = parse_dt(row.timestamp)
        if not timestamp:
            continue
        key = timestamp.date().isoformat()
        if key not in daily:
            continue
        daily[key]["total"] += 1
        daily[key][status.lower()] += 1
        if status == "PAID":
            daily[key]["revenue"] += price

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "days": days,
        "daily": list(daily.values()),
        "status_counts": status_counts,
        "vehicle_types": dict(sorted(vehicle_types.items(), key=lambda item: item[1], reverse=True)),
    }


@app.post("/api/vehicle/{plate}/upgrade")
def upgrade_vehicle(plate: str, data: PackageUpdate, db: Session = Depends(get_db), current: dict = Depends(get_current_user)):
    plate = normalize_plate(plate)
    package = normalize_package(data.package)
    rows = db.query(Detection).filter(Detection.plate_text == plate, Detection.deleted_at.is_(None)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    old_status = rows[0].payment_status
    for row in rows:
        row.payment_status = "PAID"
        row.package_type = package
        row.blacklist_reason = None
    db.add(DetectionHistory(detection_id=rows[0].id, plate_text=plate, old_status=old_status, new_status="PAID", changed_by=current["sub"]))
    log_action(db, current["sub"], "UPGRADE", detail=f"{plate} -> {package}")
    db.commit()
    return {"status": "success"}


@app.post("/api/vehicle/{plate}/cancel")
def cancel_package(plate: str, db: Session = Depends(get_db), current: dict = Depends(get_current_user)):
    plate = normalize_plate(plate)
    rows = db.query(Detection).filter(Detection.plate_text == plate, Detection.deleted_at.is_(None)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    old_status = rows[0].payment_status
    for row in rows:
        row.payment_status = "UNPAID"
        row.package_type = "STANDARD"
        row.blacklist_reason = None
    db.add(DetectionHistory(detection_id=rows[0].id, plate_text=plate, old_status=old_status, new_status="UNPAID", changed_by=current["sub"]))
    log_action(db, current["sub"], "CANCEL_PACKAGE", detail=plate)
    db.commit()
    return {"status": "success"}


@app.post("/api/vehicle/{plate}/blacklist")
def blacklist_vehicle(
    plate: str,
    data: BlacklistUpdate,
    db: Session = Depends(get_db),
    current: dict = Depends(require_role(["admin", "operator"])),
):
    plate = normalize_plate(plate)
    reason = (data.reason or "Manual review").strip()[:255]
    rows = db.query(Detection).filter(Detection.plate_text == plate, Detection.deleted_at.is_(None)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    old_status = rows[0].payment_status
    for row in rows:
        row.payment_status = "BLACKLISTED"
        row.blacklist_reason = reason
    db.add(DetectionHistory(detection_id=rows[0].id, plate_text=plate, old_status=old_status, new_status="BLACKLISTED", changed_by=current["sub"]))
    db.add(Alert(level="HIGH", message=f"Blacklisted vehicle: {plate}", plate=plate))
    log_action(db, current["sub"], "BLACKLIST", detail=f"{plate}: {reason}")
    db.commit()
    return {"status": "success"}


@app.post("/api/vehicle/{plate}/clear-blacklist")
def clear_blacklist_vehicle(
    plate: str,
    db: Session = Depends(get_db),
    current: dict = Depends(require_role(["admin", "operator"])),
):
    plate = normalize_plate(plate)
    rows = db.query(Detection).filter(Detection.plate_text == plate, Detection.deleted_at.is_(None)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    old_status = rows[0].payment_status
    for row in rows:
        if row.payment_status == "BLACKLISTED":
            row.payment_status = "UNPAID"
        row.blacklist_reason = None
    db.add(DetectionHistory(detection_id=rows[0].id, plate_text=plate, old_status=old_status, new_status=rows[0].payment_status, changed_by=current["sub"]))
    log_action(db, current["sub"], "CLEAR_BLACKLIST", detail=plate)
    db.commit()
    return {"status": "success"}


@app.delete("/api/vehicle/{plate}")
def delete_vehicle(plate: str, db: Session = Depends(get_db), current: dict = Depends(require_role(["admin", "operator"]))):
    plate = normalize_plate(plate)
    rows = db.query(Detection).filter(Detection.plate_text == plate, Detection.deleted_at.is_(None)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    for row in rows:
        row.deleted_at = datetime.utcnow()
    log_action(db, current["sub"], "SOFT_DELETE", detail=plate)
    db.commit()
    return {"status": "success"}


@app.post("/api/vehicle/{plate}/restore")
def restore_vehicle(plate: str, db: Session = Depends(get_db), current: dict = Depends(require_role(["admin"]))):
    plate = normalize_plate(plate)
    rows = db.query(Detection).filter(Detection.plate_text == plate, Detection.deleted_at.isnot(None)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Vehicle not found in trash")

    for row in rows:
        row.deleted_at = None
    log_action(db, current["sub"], "RESTORE", detail=plate)
    db.commit()
    return {"status": "success"}


@app.get("/api/trash")
def get_trash(db: Session = Depends(get_db), _: dict = Depends(require_role(["admin"]))):
    rows = db.query(Detection).filter(Detection.deleted_at.isnot(None)).order_by(desc(Detection.deleted_at), desc(Detection.id)).limit(100).all()
    return [
        {
            "id": row.id,
            "plate_text": row.plate_text,
            "vehicle_type": row.vehicle_type,
            "deleted_at": iso_dt(row.deleted_at),
            "payment_status": row.payment_status,
        }
        for row in rows
    ]


@app.get("/api/config")
def get_config_all(db: Session = Depends(get_db), _: dict = Depends(require_role(["admin"]))):
    rows = db.query(SystemConfig).order_by(SystemConfig.key).all()
    return [{"key": row.key, "value": row.value, "updated_by": row.updated_by, "updated_at": iso_dt(row.updated_at)} for row in rows]


@app.put("/api/config/{key}")
def update_config(key: str, data: ConfigUpdate, db: Session = Depends(get_db), current: dict = Depends(require_role(["admin"]))):
    if key not in CONFIG_KEYS:
        raise HTTPException(status_code=400, detail="This config key is not editable from the dashboard")

    value = data.value
    if key == "system_name":
        value = str(value).strip()
        if not value or len(value) > 80:
            raise HTTPException(status_code=400, detail="System name must be 1-80 characters")
    elif key == "price_per_vehicle":
        try:
            value = str(max(0, int(value)))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Price must be a positive integer") from exc
    elif key == "token_expire_h":
        try:
            hours = int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Token expiry must be an integer") from exc
        if hours < 1 or hours > 168:
            raise HTTPException(status_code=400, detail="Token expiry must be between 1 and 168 hours")
        value = str(hours)

    config = db.query(SystemConfig).filter(SystemConfig.key == key).first()
    if not config:
        raise HTTPException(status_code=404, detail="Config key not found")
    config.value = str(value)
    config.updated_by = current["sub"]
    config.updated_at = datetime.utcnow()
    log_action(db, current["sub"], "UPDATE_CONFIG", detail=key)
    db.commit()
    return {"status": "success"}


@app.get("/api/audit")
def get_audit(db: Session = Depends(get_db), _: dict = Depends(require_role(["admin"]))):
    rows = db.query(AuditLog).order_by(desc(AuditLog.id)).limit(200).all()
    return [
        {
            "timestamp": iso_dt(row.timestamp),
            "username": row.username,
            "action": row.action,
            "detail": row.detail,
            "ip": row.ip,
        }
        for row in rows
    ]


class ConnectionManager:
    def __init__(self):
        self.active = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active:
            try:
                await connection.send_json(message)
            except RuntimeError:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_ip = get_client_ip_from_websocket(websocket)
    allowed, reason = traffic_shield.connect_ws(client_ip)
    if not allowed:
        await websocket.close(code=1008, reason=reason)
        return

    connected = False
    try:
        await manager.connect(websocket)
        connected = True
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if connected:
            manager.disconnect(websocket)
        traffic_shield.disconnect_ws(client_ip)


@app.post("/api/internal/notify")
async def notify_new_detection(payload: dict):
    # 1. Save detection to Cloud Database
    db_sess = SessionLocal()
    try:
        plate_str = payload.get("plate")
        # Check if plate is already logged recently (e.g. within last 5 seconds to avoid double logging)
        existing = (
            db_sess.query(Detection)
            .filter(Detection.plate_text == plate_str, Detection.deleted_at.is_(None))
            .order_by(desc(Detection.timestamp))
            .first()
        )
        should_log = True
        if existing:
            time_diff = datetime.utcnow() - existing.timestamp
            if time_diff.total_seconds() < 5:
                should_log = False
                
        if should_log:
            detection = Detection(
                plate_text=plate_str,
                vehicle_type=payload.get("type", "CAR"),
                confidence=payload.get("confidence", 1.0),
                payment_status=payload.get("status", "UNPAID"),
                package_type=payload.get("package", "STANDARD"),
                blacklist_reason=payload.get("blacklist_reason"),
                lane=payload.get("lane"),
                speed_kmh=payload.get("speed_kmh", 0.0),
                crop_path=payload.get("crop_path", ""),
                timestamp=datetime.strptime(payload.get("timestamp"), "%Y-%m-%d %H:%M:%S") if payload.get("timestamp") else datetime.utcnow()
            )
            db_sess.add(detection)
            db_sess.commit()
            print(f"[CLOUD SAVE] Saved external detection: {plate_str}")
    except Exception as e:
        print(f"[CLOUD SAVE ERROR] Failed to save detection: {e}")
        db_sess.rollback()
    finally:
        db_sess.close()

    # 2. Broadcast to all active websockets
    await manager.broadcast({
        "event": "new_detection",
        "data": payload
    })
    return {"status": "success"}


# --- ADVANCED AI CONTROL AND STREAMING ENDPOINTS ---

ALLOWED_MEDIA_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}
IMAGE_MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

@app.post("/api/ai/upload")
async def upload_media(file: UploadFile = File(...), current: dict = Depends(require_role(["admin", "operator"]))):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_MEDIA_EXTS:
        raise HTTPException(status_code=400, detail="Only video (.mp4, .avi) or image (.jpg, .jpeg, .png, .webp) formats are allowed")
    
    uploads_dir = os.path.join("static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    safe_name = f"{uuid.uuid4().hex}_{re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)}"
    file_path = os.path.join(uploads_dir, safe_name)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    media_type = "image" if ext in IMAGE_MEDIA_EXTS else "video"
    return {
        "status": "success", 
        "filename": safe_name, 
        "media_type": media_type,
        "path": f"static/uploads/{safe_name}"
    }


@app.get("/api/ai/videos")
def list_videos(current: dict = Depends(get_current_user)):
    uploads_dir = os.path.join("static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    files = []
    
    all_files = sorted(
        os.listdir(uploads_dir),
        key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)) if os.path.exists(os.path.join(uploads_dir, x)) else 0,
        reverse=True
    )
    
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ALLOWED_MEDIA_EXTS:
            orig_name = f.split("_", 1)[1] if "_" in f else f
            files.append({
                "filename": f,
                "original_name": orig_name,
                "media_type": "image" if ext in IMAGE_MEDIA_EXTS else "video",
                "path": f"static/uploads/{f}"
            })
    return files


@app.post("/api/ai/start")
def start_ai(source: str = Query(...), current: dict = Depends(require_role(["admin", "operator"]))):
    if engine_module.ai_engine_running:
        # Non-blocking stop of previous instance to respond in < 1ms
        if engine_module.ai_engine_thread:
            engine_module.ai_engine_thread.stop_event.set()
        engine_module.ai_engine_running = False
        engine_module.ai_engine_thread = None
        
    # Resolve source path
    if source.isdigit():
        resolved_source = int(source)
    elif source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"):
        resolved_source = source
    else:
        # Check in static/uploads or main folder
        uploads_path = os.path.join("static", "uploads", source)
        if os.path.exists(uploads_path):
            resolved_source = uploads_path
        elif os.path.exists(source):
            resolved_source = source
        else:
            raise HTTPException(status_code=404, detail=f"Source media not found: {source}")
            
    # --- INSTANT-START OPTIMIZATION ---
    try:
        import cv2
        import numpy as np
        loading_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        loading_img[:] = (10, 7, 5)
        cv2.rectangle(loading_img, (40, 40), (1240, 680), (33, 27, 21), 2)
        cv2.putText(loading_img, "KBN TRAFFIC-AI ENGINE v6.5", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (191, 212, 45), 3)
        
        is_img = isinstance(resolved_source, str) and os.path.splitext(resolved_source)[1].lower() in IMAGE_MEDIA_EXTS
        loading_title = "ANALYZING IMAGE WITH YOLO & OCR..." if is_img else "CONNECTING TO CAMERA BOT..."
        cv2.putText(loading_img, loading_title, (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if isinstance(resolved_source, int):
            src_text = "Source: Local computer camera (Port 0)" if resolved_source == 0 else "Source: Mobile Camera / Auxiliary Cam (Port 1)"
        else:
            clean_display = source.split("_", 1)[1] if "_" in source else source
            src_text = f"Source: {clean_display}"
        cv2.putText(loading_img, src_text, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (184, 163, 148), 2)
        cv2.putText(loading_img, "Please wait, starting YOLOv8 + OpenVINO...", (100, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (94, 197, 34), 2)
        
        _, jpeg = cv2.imencode('.jpg', loading_img)
        with latest_frame_lock:
            engine_module.latest_processed_frame = jpeg.tobytes()
    except Exception as ex:
        print(f"Error drawing loading frame: {ex}")
        
        if isinstance(resolved_source, int):
            src_text = "Source: Local computer camera (Port 0)" if resolved_source == 0 else "Source: Mobile Camera / Auxiliary Cam (Port 1)"
        else:
            src_text = f"Source: {source}"
        cv2.putText(loading_img, src_text, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (184, 163, 148), 2)
        cv2.putText(loading_img, "Please wait, starting YOLOv8 + OpenVINO...", (100, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (94, 197, 34), 2)
        
        _, jpeg = cv2.imencode('.jpg', loading_img)
        with latest_frame_lock:
            engine_module.latest_processed_frame = jpeg.tobytes()
    except Exception as ex:
        print(f"Error drawing loading frame: {ex}")
    # ----------------------------------

    # Start the engine thread
    engine_module.ai_engine_thread = TrafficAIEngine(resolved_source)
    engine_module.ai_engine_running = True
    engine_module.ai_engine_thread.start()
    
    return {"status": "success", "message": f"Started AI Engine on {source}"}


@app.post("/api/ai/stop")
def stop_ai(current: dict = Depends(require_role(["admin", "operator"]))):
    if not engine_module.ai_engine_running or engine_module.ai_engine_thread is None:
        return {"status": "success", "message": "AI engine was not running"}
        
    # Stop thread
    engine_module.ai_engine_thread.stop_event.set()
    engine_module.ai_engine_thread.join(timeout=2.0)
    
    engine_module.ai_engine_running = False
    engine_module.ai_engine_thread = None
    with latest_frame_lock:
        engine_module.latest_processed_frame = None
        
    return {"status": "success", "message": "Stopped AI Engine"}


@app.get("/api/ai/status")
def get_ai_status(current: dict = Depends(get_current_user)):
    if not engine_module.ai_engine_running or engine_module.ai_engine_thread is None:
        return {
            "running": False,
            "status": "Idle",
            "fps": 0,
            "processed_frames": 0
        }
        
    return {
        "running": True,
        "status": engine_module.ai_engine_thread.status,
        "fps": engine_module.ai_engine_thread.fps,
        "processed_frames": engine_module.ai_engine_thread.processed_frames
    }


@app.get("/api/stream")
def video_stream():
    def frame_generator():
        last_sent_frame = None
        while engine_module.ai_engine_running:
            frame = None
            with latest_frame_lock:
                frame = engine_module.latest_processed_frame
            if frame is not None and frame != last_sent_frame:
                last_sent_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.03)
        yield b''
            
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/reports/charts")
def get_reports_charts(
    days: int = Query(14, ge=1, le=90),
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_user),
):
    price = get_price_per_vehicle(db)
    start_date = (datetime.utcnow() - timedelta(days=days - 1)).date()
    daily = {
        (start_date + timedelta(days=offset)).isoformat(): {
            "date": (start_date + timedelta(days=offset)).isoformat(),
            "total": 0,
            "paid": 0,
            "unpaid": 0,
            "blacklisted": 0,
            "revenue": 0,
        }
        for offset in range(days)
    }
    
    vehicle_types = {"CAR": 0, "TRUCK": 0, "BUS": 0}
    
    rows = db.query(Detection).filter(Detection.deleted_at.is_(None)).all()
    for row in rows:
        status = row.payment_status if row.payment_status in VALID_STATUSES else "UNPAID"
        v_type = (row.vehicle_type or "CAR").upper()
        if v_type in vehicle_types:
            vehicle_types[v_type] += 1
        else:
            vehicle_types[v_type] = vehicle_types.get(v_type, 0) + 1
            
        timestamp = parse_dt(row.timestamp)
        if not timestamp:
            continue
        key = timestamp.date().isoformat()
        if key not in daily:
            continue
        daily[key]["total"] += 1
        daily[key][status.lower()] += 1
        if status == "PAID":
            daily[key]["revenue"] += price
            
    sorted_vehicle_types = dict(sorted(vehicle_types.items(), key=lambda item: item[1], reverse=True))
    
    return {
        "days": days,
        "daily": list(daily.values()),
        "vehicle_types": sorted_vehicle_types
    }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", context={})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        limit_concurrency=settings.UVICORN_LIMIT_CONCURRENCY,
        timeout_keep_alive=settings.UVICORN_TIMEOUT_KEEP_ALIVE,
        proxy_headers=settings.TRUST_PROXY_HEADERS,
    )
