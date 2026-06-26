"""
image_uploader.py — rate-limited JPEG frame uploader to S3.

Called from the main publish loop.  Credentials come from the Greengrass
TokenExchangeService — the component recipe must declare:
    aws.greengrass.TokenExchangeService as a HARD dependency.

Without that dependency the component's IAM role credentials are not
injected into the environment and boto3 will fail with auth errors.
"""

from __future__ import annotations

import logging
import os
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
        Key prefix (e.g. "snapshots"). Final key: ``{prefix}/{device_id}/{unix_ts_ms}.jpg``.
    min_interval_s:
        Minimum seconds between uploads.
    jpeg_quality:
        JPEG quality 1-100. Default 75.
    region:
        AWS region of the bucket (e.g. "ap-south-1").  Required for buckets
        outside us-east-1 — boto3 will fail with signature errors otherwise.
        Falls back to AWS_DEFAULT_REGION env var if empty.
    always_upload:
        When True, upload on every interval tick even with no detections.
        When False (default), only upload when has_detections=True.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        min_interval_s: float = 5.0,
        jpeg_quality: int = 75,
        region: str = "",
        always_upload: bool = False,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._min_interval_s = min_interval_s
        self._jpeg_quality = jpeg_quality
        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "") or os.environ.get("AWS_REGION", "")
        self._always_upload = always_upload
        self._last_upload_ts: float = 0.0
        self._client: Optional[object] = None
        self._upload_count: int = 0

        if not bucket:
            logger.warning(
                "ImageUploader: no S3 bucket configured — uploads will be skipped. "
                "Set image_upload.s3_bucket in config.yaml or AWS_S3_BUCKET env var."
            )
        else:
            logger.info(
                "ImageUploader: ready — bucket=%s prefix=%s region=%s interval=%.1fs "
                "quality=%d always_upload=%s",
                bucket,
                prefix,
                self._region or "(boto3 default)",
                min_interval_s,
                jpeg_quality,
                always_upload,
            )
            if not self._region:
                logger.warning(
                    "ImageUploader: no region set — boto3 will default to us-east-1. "
                    "If your bucket is in another region set image_upload.s3_region in config.yaml."
                )

    def set_always_upload(self, value: bool) -> None:
        self._always_upload = value
        logger.info("ImageUploader: always_upload set to %s via remote command", value)

    def _get_client(self):
        if self._client is None:
            import boto3  # type: ignore[import]
            kwargs: dict = {}
            if self._region:
                kwargs["region_name"] = self._region
            self._client = boto3.client("s3", **kwargs)
            logger.info(
                "ImageUploader: boto3 S3 client created (region=%s)",
                self._region or "boto3-default",
            )
        return self._client

    def maybe_upload(
        self,
        frame: np.ndarray,
        device_id: str,
        has_detections: bool,
    ) -> Optional[str]:
        """
        Upload *frame* as a JPEG to S3 if the rate limit has not been hit.

        Uploads when has_detections=True OR always_upload=True.
        Returns the S3 key on upload, or None if the upload was skipped.
        """
        if not self._always_upload and not has_detections:
            return None

        now = time.time()
        if now - self._last_upload_ts < self._min_interval_s:
            return None

        if not self._bucket:
            logger.warning(
                "ImageUploader: bucket not configured — cannot upload. "
                "Set image_upload.s3_bucket or AWS_S3_BUCKET env var."
            )
            return None

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok or buf is None:
                logger.error("ImageUploader: JPEG encoding failed")
                return None

            ts_ms = int(now * 1000)
            key = f"{self._prefix}/{device_id}/{ts_ms}.jpg"

            logger.info(
                "ImageUploader: uploading s3://%s/%s (%d bytes, detections=%s)...",
                self._bucket,
                key,
                len(buf),
                has_detections,
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
                "ImageUploader: OK — s3://%s/%s total_uploads=%d",
                self._bucket,
                key,
                self._upload_count,
            )
            return key

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "ImageUploader: upload FAILED — %s\n"
                "  Check: (1) recipe has aws.greengrass.TokenExchangeService HARD dependency,\n"
                "         (2) Greengrass token exchange IAM role has s3:PutObject on the bucket,\n"
                "         (3) image_upload.s3_region matches the bucket's actual region.",
                exc,
                exc_info=True,
            )
            return None
