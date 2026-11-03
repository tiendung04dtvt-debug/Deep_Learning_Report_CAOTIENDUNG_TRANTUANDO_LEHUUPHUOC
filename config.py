from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):





    SECRET_KEY: str = "AI_TRAFFIC_ULTRA_PREMIUM_SECURE_TOKEN_2026_LONG"
    ALGORITHM: str = "HS256"
    TOKEN_EXPIRE_H: int = 8


    DATABASE_URL: str = "sqlite:///./traffic_monitoring.db"


    PRICE_PER_VEHICLE: int = 50000
    SYSTEM_NAME: str = "TrafficAI"
    VERSION: str = "6.5.0"


    ALLOWED_ORIGINS: list[str] = ["http://localhost:8001", "http://127.0.0.1:8001"]


    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 600
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    LOGIN_RATE_LIMIT_REQUESTS: int = 15
    LOGIN_RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_BAN_SECONDS: int = 0
    MAX_REQUEST_BODY_BYTES: int = 200_000_000
    MAX_WS_PER_IP: int = 3
    MAX_WS_TOTAL: int = 50
    TRUST_PROXY_HEADERS: bool = False
    UVICORN_LIMIT_CONCURRENCY: int = 120
    UVICORN_TIMEOUT_KEEP_ALIVE: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
