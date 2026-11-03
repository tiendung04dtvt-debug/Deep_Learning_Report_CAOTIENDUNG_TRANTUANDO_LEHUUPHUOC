import time
from collections import defaultdict, deque
from threading import Lock
from config import settings

class TrafficShield:


    def __init__(self):
        self.lock = Lock()
        self.buckets = defaultdict(deque)
        self.bans: dict[str, float] = {}
        self.ws_by_ip = defaultdict(int)
        self.ws_total = 0
        self.allowed_requests = 0
        self.blocked_requests = 0
        self.body_rejected = 0
        self.websocket_rejected = 0

    def _cleanup_bucket(self, values: deque, now: float, window_seconds: int):
        cutoff = now - window_seconds
        while values and values[0] <= cutoff:
            values.popleft()

    def _limit_for_path(self, path: str) -> tuple[str, int, int]:
        if path == "/api/login":
            return "login", settings.LOGIN_RATE_LIMIT_REQUESTS, settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS
        return "global", settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS

    def check_http(self, ip: str, path: str, method: str, content_length: str | None) -> tuple[bool, int, str, int]:
        if not settings.RATE_LIMIT_ENABLED or method == "OPTIONS":
            return True, 200, "", 0


        if path in ("/api/ai/status", "/api/stream", "/api/health") or path.startswith("/static/"):
            return True, 200, "", 0


        if ip in ("127.0.0.1", "::1", "localhost") and path != "/api/login":
            return True, 200, "", 0

        max_body = max(0, settings.MAX_REQUEST_BODY_BYTES)
        if content_length and max_body:
            try:
                if int(content_length) > max_body:
                    with self.lock:
                        self.blocked_requests += 1
                        self.body_rejected += 1
                    return False, 413, "Request body is too large", 0
            except ValueError:
                with self.lock:
                    self.blocked_requests += 1
                    self.body_rejected += 1
                return False, 400, "Invalid Content-Length header", 0

        now = time.monotonic()
        with self.lock:
            ban_until = self.bans.get(ip)
            if ban_until:
                if ban_until > now:
                    self.blocked_requests += 1
                    return False, 429, "IP is temporarily rate limited", max(1, int(ban_until - now))
                del self.bans[ip]

            bucket_name, limit, window = self._limit_for_path(path)
            key = (ip, bucket_name)
            values = self.buckets[key]
            self._cleanup_bucket(values, now, window)
            if len(values) >= max(1, limit):
                ban_seconds = max(0, settings.RATE_LIMIT_BAN_SECONDS)
                if ban_seconds:
                    self.bans[ip] = now + ban_seconds
                self.blocked_requests += 1
                retry_after = ban_seconds or max(1, int(window - (now - values[0])))
                return False, 429, "Too many requests", retry_after

            values.append(now)
            self.allowed_requests += 1
            return True, 200, "", 0

    def connect_ws(self, ip: str) -> tuple[bool, str]:
        if not settings.RATE_LIMIT_ENABLED or ip in ("127.0.0.1", "::1", "localhost"):
            return True, ""

        now = time.monotonic()
        with self.lock:
            ban_until = self.bans.get(ip)
            if ban_until:
                if ban_until > now:
                    self.websocket_rejected += 1
                    return False, "IP is temporarily rate limited"
                del self.bans[ip]

            if self.ws_total >= max(1, settings.MAX_WS_TOTAL):
                self.websocket_rejected += 1
                return False, "WebSocket capacity reached"
            if self.ws_by_ip[ip] >= max(1, settings.MAX_WS_PER_IP):
                self.websocket_rejected += 1
                return False, "Too many WebSocket connections from this IP"

            self.ws_total += 1
            self.ws_by_ip[ip] += 1
            return True, ""

    def disconnect_ws(self, ip: str):
        with self.lock:
            if self.ws_by_ip[ip] > 0:
                self.ws_by_ip[ip] -= 1
                self.ws_total = max(0, self.ws_total - 1)
            if self.ws_by_ip[ip] == 0:
                self.ws_by_ip.pop(ip, None)

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            expired = [ip for ip, until in self.bans.items() if until <= now]
            for ip in expired:
                del self.bans[ip]
            return {
                "enabled": settings.RATE_LIMIT_ENABLED,
                "global_limit": {
                    "requests": settings.RATE_LIMIT_REQUESTS,
                    "window_seconds": settings.RATE_LIMIT_WINDOW_SECONDS,
                },
                "login_limit": {
                    "requests": settings.LOGIN_RATE_LIMIT_REQUESTS,
                    "window_seconds": settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS,
                },
                "ban_seconds": settings.RATE_LIMIT_BAN_SECONDS,
                "max_request_body_bytes": settings.MAX_REQUEST_BODY_BYTES,
                "active_bans": len(self.bans),
                "active_websockets": self.ws_total,
                "active_websockets_by_ip": dict(self.ws_by_ip),
                "allowed_requests": self.allowed_requests,
                "blocked_requests": self.blocked_requests,
                "body_rejected": self.body_rejected,
                "websocket_rejected": self.websocket_rejected,
            }
