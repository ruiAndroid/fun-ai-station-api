import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_env_file() -> Optional[str]:
    """
    Preferred: set ENV_FILE to point to a local env file (e.g. local.env).
    Fallback: .env if exists.
    """
    env_file = os.getenv("ENV_FILE")
    if env_file:
        return env_file
    if os.path.exists(".env"):
        return ".env"
    if os.path.exists("local.env"):
        return "local.env"
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_get_env_file(), env_file_encoding="utf-8")

    # Server
    APP_NAME: str = "fun-ai-station-api"
    ENV: str = "local"
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Database
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str = "db_funaistation"
    DB_USER: str = "root"
    DB_PASSWORD: str = ""

    # Auth
    JWT_SECRET: str = "change_me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRES_MINUTES: int = 60

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

