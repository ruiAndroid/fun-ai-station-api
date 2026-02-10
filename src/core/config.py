from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional


def _repo_root() -> Path:
    # This file is at: fun-ai-station-api/src/core/config.py
    return Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = _repo_root() / "configs" / "fun-ai-station-api.env"


def _parse_env_file(content: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in (content or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        out[key] = val
    return out


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Please create it based on configs/fun-ai-station-api.env."
        )
    return _parse_env_file(path.read_text(encoding="utf-8"))


def _get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, "").strip() or default)
    except Exception:
        return default


class Settings:
    """
    Read settings ONLY from `configs/fun-ai-station-api.env`.
    Do not depend on process environment variables.
    """

    # Server
    APP_NAME: str
    ENV: str
    HOST: str
    PORT: int

    # Database
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # Auth
    JWT_SECRET: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRES_MINUTES: int

    # Agent service (Node + Python runtime)
    FUN_AGENT_SERVICE_URL: str

    # Openclaw webhook (receive messages from another server)
    OPENCLAW_WEBHOOK_SECRET: str
    OPENCLAW_MAX_SKEW_SECONDS: int
    OPENCLAW_DEFAULT_AGENT: str

    # OpenAI-compatible API (for OpenClaw to call as "LLM backend")
    OPENAI_API_KEY: str
    OPENAI_DEFAULT_AGENT: str

    def __init__(self, cfg: Optional[Dict[str, str]] = None):
        cfg = cfg or _load_config()
        self.APP_NAME = cfg.get("APP_NAME", "fun-ai-station-api")
        self.ENV = cfg.get("ENV", "prod")
        self.HOST = cfg.get("HOST", "0.0.0.0")
        self.PORT = _get_int(cfg, "PORT", 8001)

        self.DB_HOST = cfg.get("DB_HOST", "127.0.0.1")
        self.DB_PORT = _get_int(cfg, "DB_PORT", 3306)
        self.DB_NAME = cfg.get("DB_NAME", "db_funaistation")
        self.DB_USER = cfg.get("DB_USER", "root")
        self.DB_PASSWORD = cfg.get("DB_PASSWORD", "")

        self.JWT_SECRET = cfg.get("JWT_SECRET", "change_me")
        self.JWT_ALGORITHM = cfg.get("JWT_ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRES_MINUTES = _get_int(cfg, "ACCESS_TOKEN_EXPIRES_MINUTES", 60)

        self.FUN_AGENT_SERVICE_URL = cfg.get("FUN_AGENT_SERVICE_URL", "http://127.0.0.1:4010")

        # Openclaw webhook
        self.OPENCLAW_WEBHOOK_SECRET = cfg.get("OPENCLAW_WEBHOOK_SECRET", "")
        self.OPENCLAW_MAX_SKEW_SECONDS = _get_int(cfg, "OPENCLAW_MAX_SKEW_SECONDS", 300)
        self.OPENCLAW_DEFAULT_AGENT = cfg.get("OPENCLAW_DEFAULT_AGENT", "attendance")

        # OpenAI-compatible API
        self.OPENAI_API_KEY = cfg.get("OPENAI_API_KEY", "")
        self.OPENAI_DEFAULT_AGENT = cfg.get("OPENAI_DEFAULT_AGENT", self.OPENCLAW_DEFAULT_AGENT or "attendance")

    @property
    def sqlalchemy_database_uri(self) -> str:
        # mysql+pymysql://user:pass@host:port/db?charset=utf8mb4
        user = self.DB_USER
        password = self.DB_PASSWORD
        host = self.DB_HOST
        port = self.DB_PORT
        db = self.DB_NAME
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

