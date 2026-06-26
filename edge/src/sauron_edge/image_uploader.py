"""
image_uploader.py — rate-limited JPEG frame uploader to S3.

Called from the main publish loop whenever stable detections are present
and enough time has passed since the last upload.  Credentials come from
the Greengrass component's IAM role — no extra configuration required on
the edge device itself.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageUploader:
    """
    Uploads JPEG-encoded frames to S3, rate-limited to at most one upload
    per *min_interval_s* seconds.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    prefix:
        Key prefix (e.g. "snapshots"). The final key is
        ``{prefix}/{device_id}/{unix_ts_ms}.jpg``.
    min_interval_s:
        Minimum seconds between uploads. Upload is skipped if not enough
        time has elapsed since the last successful upload.
    jpeg_quality:
        JPEG quality 1-100. Default 75.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        min_interval_s: float = 5.0,
        jpeg_quality: int = 75,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._min_interval_s = min_interval_s
        self._jpeg_quality = jpeg_quality
        self._last_upload_ts: float = 0.0
        self._client: Optional[object] = None

    def _get_client(self):
        if self._client is None:
            import boto3  # type: ignore[import]
            self._client = boto3.client("s3")
        return self._client

    def maybe_upload(
        self,
        frame: np.ndarray,
        device_id: str,
        has_detections: bool,
    ) -> Optional[str]:
        """
        Upload *frame* as a JPEG to S3 if detections are present and the
        rate limit has not been hit.

        Returns the S3 key on upload, or None if the upload was skipped.
        """
        if not has_detections:
            return None

        now = time.time()
        if now - self._last_upload_ts < self._min_interval_s:
            return None

        if not self._bucket:
            logger.debug("ImageUploader: no bucket configured — skipping upload")
            return None

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok or buf is None:
                logger.warning("ImageUploader: JPEG encoding failed")
                return None

            ts_ms = int(now * 1000)
            key = f"{self._prefix}/{device_id}/{ts_ms}.jpg"

            self._get_client().put_object(
                Bucket=self._bucket,
                Key=key,
                Body=buf.tobytes(),
                ContentType="image/jpeg",
            )

            self._last_upload_ts = now
            logger.info("ImageUploader: uploaded %s (%d bytes)", key, len(buf))
            return key

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("ImageUploader: upload failed — %s", exc)
            return None
