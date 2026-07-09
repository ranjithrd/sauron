from typing import List

from pydantic_settings import BaseSettings


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

    # Triangulation confidence tuning — how "fuzzy" ray-convergence matching is.
    # TRIANGULATION_CONFIDENCE_SCALE_M: larger = more tolerant of ray separation
    # (confidence decays more slowly with distance between the two closest-approach points).
    # MIN_TRIANGULATION_CONFIDENCE: triangulations scoring below this are discarded.
    TRIANGULATION_CONFIDENCE_SCALE_M: float = 25.0
    MIN_TRIANGULATION_CONFIDENCE: float = 0.15

    # S3 snapshot storage
    AWS_S3_BUCKET: str = ""
    AWS_REGION: str = "ap-south-1"
    AWS_ACCESS_KEY_ID: str = ""  # optional; falls back to instance profile / env
    AWS_SECRET_ACCESS_KEY: str = ""

    # VLM inference via OpenRouter (litellm)
    # Set VLM_API_KEY to your OpenRouter API key (sk-or-...)
    # Override VLM_MODEL with any openrouter/<provider>/<model> string
    VLM_MODEL: str = "openrouter/qwen/qwen3-vl-8b-instruct"
    VLM_API_KEY: str = ""
    VLM_INTERVAL_S: int = 60
    SUMMARY_MODEL: str = "openrouter/qwen/qwen3-8b"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
