from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://iotel:iotel@localhost:5432/iotel"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    DETECTION_CORRELATION_WINDOW_MS: int = 200
    OBJECT_STALE_TIMEOUT_S: float = 30.0
    FRAME_SKIP: int = 2
    IOT_ENDPOINT: str = ""
    IOT_CERT_PATH: str = "certs/certificate.pem.crt"
    IOT_KEY_PATH: str = "certs/private.pem.key"
    IOT_CA_PATH: str = "certs/AmazonRootCA1.pem"
    IOT_CLIENT_ID: str = "iotel-server"
    CAMERA_HEIGHT_M: float = 2.0
    ASSOCIATION_MAX_DIST_M: float = 200.0

    class Config:
        env_file = ".env"


settings = Settings()
