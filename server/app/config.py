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

    # S3 snapshot storage
    AWS_S3_BUCKET: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""       # optional; falls back to instance profile / env
    AWS_SECRET_ACCESS_KEY: str = ""

    # VLM inference (litellm — supports any provider)
    VLM_MODEL: str = "gpt-4o"        # litellm model string, e.g. claude-3-5-sonnet-20241022
    VLM_API_KEY: str = ""            # provider API key
    VLM_INTERVAL_S: int = 30         # seconds between automatic VLM calls when enabled

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
