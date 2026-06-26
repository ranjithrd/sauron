"""
image_uploader.py — rate-limited JPEG frame uploader to S3.

Called from the main publish loop whenever stable detections are present
and enough time has passed since the last upload.  Credentials come from
the Greengrass component's IAM role — no extra configuration required on
the edge device itself.
"""

from __future__ import annotations

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
        Minimum seconds between uploads.
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
        self._upload_count: int = 0
        self._skip_no_detect: int = 0
        self._skip_rate_limit: int = 0

        if not bucket:
            logger.warning(
                "ImageUploader: no S3 bucket configured — uploads will be skipped "
                "until bucket is set via image_upload.s3_bucket or AWS_S3_BUCKET env var"
            )
        else:
            logger.info(
                "ImageUploader: ready — bucket=%s prefix=%s interval=%.1fs quality=%d",
                bucket,
                prefix,
                min_interval_s,
                jpeg_quality,
            )

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
            self._skip_no_detect += 1
            if self._skip_no_detect % 100 == 1:
                logger.debug(
                    "ImageUploader: skipping upload — no stable detections "
                    "(skipped %d times so far; uploads only happen when objects are detected)",
                    self._skip_no_detect,
                )
            return None

        now = time.time()
        elapsed = now - self._last_upload_ts
        if elapsed < self._min_interval_s:
            self._skip_rate_limit += 1
            return None

        if not self._bucket:
            logger.warning(
                "ImageUploader: bucket not configured — cannot upload snapshot. "
                "Set image_upload.s3_bucket in config.yaml or AWS_S3_BUCKET env var."
            )
            return None

        try:
            logger.info(
                "ImageUploader: encoding frame for device=%s (upload #%d)",
                device_id,
                self._upload_count + 1,
            )
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok or buf is None:
                logger.error("ImageUploader: JPEG encoding failed")
                return None

            ts_ms = int(now * 1000)
            key = f"{self._prefix}/{device_id}/{ts_ms}.jpg"

            logger.info(
                "ImageUploader: uploading to s3://%s/%s (%d bytes)...",
                self._bucket,
                key,
                len(buf),
            )
            self._get_client().put_object(
                Bucket=self._bucket,
                Key=key,
                Body=buf.tobytes(),
                ContentType="image/jpeg",
            )

            self._last_upload_ts = now
            self._upload_count += 1
            logger.info(
                "ImageUploader: uploaded s3://%s/%s (%d bytes) — total_uploads=%d",
                self._bucket,
                key,
                len(buf),
                self._upload_count,
            )
            return key

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "ImageUploader: upload FAILED — %s. "
                "Check: (1) IAM role has s3:PutObject on the bucket, "
                "(2) bucket name is correct, "
                "(3) AWS region matches the bucket.",
                exc,
                exc_info=True,
            )
            return None
