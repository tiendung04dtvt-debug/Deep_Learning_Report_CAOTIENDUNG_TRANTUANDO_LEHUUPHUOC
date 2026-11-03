from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from src.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="operator")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ip = Column(String)
    user_agent = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    plate_text = Column(String, index=True)
    vehicle_type = Column(String)
    confidence = Column(Float)
    payment_status = Column(String, default="UNPAID")
    package_type = Column(String, default="STANDARD")
    blacklist_reason = Column(String, nullable=True)
    lane = Column(String, nullable=True)
    speed_kmh = Column(Float, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    crop_path = Column(String, nullable=True)
    direction = Column(String, nullable=True)


class DetectionHistory(Base):
    __tablename__ = "detection_history"
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    plate_text = Column(String)
    old_status = Column(String)
    new_status = Column(String)
    changed_by = Column(String)
    changed_at = Column(DateTime, default=datetime.utcnow)


class SystemConfig(Base):
    __tablename__ = "system_config"
    key = Column(String, primary_key=True, unique=True)
    value = Column(String)
    updated_by = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)


class DailySummary(Base):
    __tablename__ = "daily_summaries"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, unique=True)
    total = Column(Integer)
    paid = Column(Integer)
    blacklisted = Column(Integer)
    revenue = Column(Integer)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pref_key = Column(String)
    pref_value = Column(String)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String)
    message = Column(String)
    plate = Column(String, nullable=True)
    resolved = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    username = Column(String)
    action = Column(String)
    detail = Column(String)
    ip = Column(String)
