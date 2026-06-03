import os
from pathlib import Path

import boto3
from botocore.client import Config


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")
BATCHES_BUCKET = os.getenv("MINIO_BATCHES_BUCKET", "flocktory-batches")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_bytes_to_minio(
    file_bytes: bytes,
    object_key: str,
    content_type: str,
    bucket: str = BATCHES_BUCKET,
) -> str:
    s3 = get_s3_client()

    s3.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=file_bytes,
        ContentType=content_type,
    )

    return f"s3://{bucket}/{object_key}"


def list_bucket_objects(bucket: str = BATCHES_BUCKET) -> list[str]:
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket)

    if "Contents" not in response:
        return []

    return [obj["Key"] for obj in response["Contents"]]

