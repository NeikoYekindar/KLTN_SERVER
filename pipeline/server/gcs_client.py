"""
GCS Client — upload/download helpers for Google Cloud Storage.
Set GCS_BUCKET env var to enable; if unset, all operations are no-ops.
"""

import json
import os

GCS_BUCKET = os.environ.get('GCS_BUCKET', '')

_client = None


def _get_client():
    global _client
    if _client is None:
        from google.cloud import storage
        _client = storage.Client()
    return _client


def _enabled():
    if not GCS_BUCKET:
        return False
    try:
        _get_client()
        return True
    except Exception as e:
        print(f"[GCS] Client init failed: {e}")
        return False


def upload_file(local_path, gcs_path: str, bucket_name: str = None):
    """Upload a local file to GCS."""
    bucket_name = bucket_name or GCS_BUCKET
    if not bucket_name:
        return
    try:
        bucket = _get_client().bucket(bucket_name)
        bucket.blob(gcs_path).upload_from_filename(str(local_path))
        print(f"[GCS] ↑ gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"[GCS] Upload failed ({gcs_path}): {e}")


def upload_string(content: str, gcs_path: str, bucket_name: str = None, content_type: str = 'text/plain'):
    """Upload a string directly to GCS (avoids a local temp file)."""
    bucket_name = bucket_name or GCS_BUCKET
    if not bucket_name:
        return
    try:
        bucket = _get_client().bucket(bucket_name)
        bucket.blob(gcs_path).upload_from_string(content, content_type=content_type)
        print(f"[GCS] ↑ gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"[GCS] Upload failed ({gcs_path}): {e}")


def download_json(gcs_path: str, bucket_name: str = None):
    """Download and parse JSON from GCS. Returns dict or None."""
    bucket_name = bucket_name or GCS_BUCKET
    if not bucket_name:
        return None
    try:
        bucket = _get_client().bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"[GCS] Download failed ({gcs_path}): {e}")
        return None


def download_text(gcs_path: str, bucket_name: str = None):
    """Download raw text (CSV, etc.) from GCS. Returns str or None."""
    bucket_name = bucket_name or GCS_BUCKET
    if not bucket_name:
        return None
    try:
        bucket = _get_client().bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            return None
        return blob.download_as_text()
    except Exception as e:
        print(f"[GCS] Download failed ({gcs_path}): {e}")
        return None


def list_blobs(prefix: str, bucket_name: str = None):
    """List blob names under a GCS prefix. Returns list[str]."""
    bucket_name = bucket_name or GCS_BUCKET
    if not bucket_name:
        return []
    try:
        blobs = _get_client().list_blobs(bucket_name, prefix=prefix)
        return [b.name for b in blobs]
    except Exception as e:
        print(f"[GCS] List failed ({prefix}): {e}")
        return []
