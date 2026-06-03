from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.queue import enqueue_batch
from app.storage import list_bucket_objects, upload_bytes_to_minio
from app.db import (
    create_batch_record,
    get_batch_by_id,
    list_batches,
    list_clusters_by_batch,
    list_predictions_by_batch,
    update_batch_status,
)


app = FastAPI(
    title="Flocktory Deduplication Backend API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_batch(file: UploadFile = File(...)):
    allowed_extensions = (".csv", ".parquet")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Parquet files are supported",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    batch_id = uuid4()
    uploaded_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    object_key = f"batches/{uploaded_at}_{batch_id}/{file.filename}"

    content_type = file.content_type or "application/octet-stream"

    s3_uri = upload_bytes_to_minio(
        file_bytes=file_bytes,
        object_key=object_key,
        content_type=content_type,
    )

    batch_record = create_batch_record(
        batch_id=batch_id,
        filename=file.filename,
        object_key=object_key,
        s3_uri=s3_uri,
        status="uploaded",
    )

    return {
        "status": "uploaded",
        "batch": batch_record,
    }


@app.get("/batches/objects")
def get_batch_objects():
    return {
        "objects": list_bucket_objects()
    }


@app.get("/batches")
def get_batches():
    return {
        "batches": list_batches()
    }


@app.get("/batches/{batch_id}/predictions")
def get_batch_predictions(batch_id: str):
    return {
        "batch_id": batch_id,
        "predictions": list_predictions_by_batch(batch_id),
    }


@app.post("/batches/{batch_id}/process")
def process_batch(batch_id: str):
    try:
        batch = get_batch_by_id(batch_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    if batch["status"] in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail=f"Batch is already {batch['status']}",
        )

    updated_batch = update_batch_status(batch_id, "queued")
    enqueue_batch(batch_id)

    return {
        "batch_id": batch_id,
        "status": updated_batch["status"],
        "message": "Batch has been queued for processing",
    }


@app.get("/batches/{batch_id}")
def get_batch(batch_id: str):
    try:
        batch = get_batch_by_id(batch_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "batch": batch
    }

@app.get("/batches/{batch_id}/clusters")
def get_batch_clusters(batch_id: str):
    return {
        "batch_id": batch_id,
        "clusters": list_clusters_by_batch(batch_id),
    }