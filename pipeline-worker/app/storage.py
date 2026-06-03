import os
from pathlib import Path

import boto3
import pandas as pd
from botocore.client import Config


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")

BATCHES_BUCKET = os.getenv("MINIO_BATCHES_BUCKET", "flocktory-batches")
ARTIFACTS_BUCKET = os.getenv("MINIO_ARTIFACTS_BUCKET", "flocktory-artifacts")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_file_to_minio(
    local_path: str | Path,
    object_key: str,
    bucket: str = BATCHES_BUCKET,
) -> str:
    """Загружает локальный файл в MinIO."""
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    s3 = get_s3_client()
    s3.upload_file(str(local_path), bucket, object_key)

    return f"s3://{bucket}/{object_key}"


def download_file_from_minio(
    object_key: str,
    local_path: str | Path,
    bucket: str = BATCHES_BUCKET,
) -> Path:
    """Скачивает объект из MinIO в локальный файл внутри контейнера."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()
    s3.download_file(bucket, object_key, str(local_path))

    return local_path


def list_bucket_objects(bucket: str = BATCHES_BUCKET) -> list[str]:
    """Возвращает список object keys в bucket."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket)

    if "Contents" not in response:
        return []

    return [obj["Key"] for obj in response["Contents"]]


def read_csv_from_minio(
    object_key: str,
    bucket: str = BATCHES_BUCKET,
) -> pd.DataFrame:
    """Скачивает CSV из MinIO во временную директорию и читает его в DataFrame."""
    local_path = Path("/tmp/flocktory-batches") / Path(object_key).name

    download_file_from_minio(
        object_key=object_key,
        local_path=local_path,
        bucket=bucket,
    )

    return pd.read_csv(local_path)

def read_batch_from_minio(
    object_key: str,
    bucket: str = BATCHES_BUCKET,
) -> pd.DataFrame:
    """Скачивает batch-файл из MinIO и читает CSV или Parquet."""
    local_path = Path("/tmp/flocktory-batches") / Path(object_key).name

    download_file_from_minio(
        object_key=object_key,
        local_path=local_path,
        bucket=bucket,
    )

    suffix = local_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(local_path)

    if suffix == ".parquet":
        return pd.read_parquet(local_path)

    raise ValueError(f"Unsupported file format: {suffix}")