from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://iotel:iotel@localhost:5432/iotel"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    DETECTION_CORRELATION_WINDOW_MS: int = 200
    OBJECT_STALE_TIMEOUT_S: float = 2.0
    FRAME_SKIP: int = 2

    class Config:
        env_file = ".env"


settings = Settings()
