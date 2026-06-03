import os
from typing import Any
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "flocktory")


def get_required_env(name: str) -> str:
    """Возвращает обязательную переменную окружения или выбрасывает ошибку."""
    value = os.getenv(name)

    if not value:
        raise ValueError(f"Environment variable {name} is required")

    return value


def get_connection():
    """Создаёт подключение к PostgreSQL."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=get_required_env("POSTGRES_USER"),
        password=get_required_env("POSTGRES_PASSWORD"),
        cursor_factory=RealDictCursor,
    )


def create_batch_record(
    batch_id: UUID,
    filename: str,
    object_key: str,
    s3_uri: str,
    status: str = "uploaded",
) -> dict[str, Any]:
    """Создаёт запись о загруженном batch-файле."""
    query = """
        INSERT INTO batches (
            batch_id,
            filename,
            object_key,
            s3_uri,
            status
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING
            batch_id,
            filename,
            object_key,
            s3_uri,
            status,
            uploaded_at,
            updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    str(batch_id),
                    filename,
                    object_key,
                    s3_uri,
                    status,
                ),
            )
            return dict(cursor.fetchone())


def list_batches() -> list[dict[str, Any]]:
    """Возвращает список загруженных batch-файлов."""
    query = """
        SELECT
            batch_id,
            filename,
            object_key,
            s3_uri,
            status,
            uploaded_at,
            updated_at
        FROM batches
        ORDER BY uploaded_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]


def list_predictions_by_batch(batch_id: str) -> list[dict[str, Any]]:
    """Возвращает predictions для конкретного batch."""
    query = """
        SELECT
            prediction_id,
            batch_id,
            profile1,
            profile2,
            match_score,
            is_duplicate,
            recommendation,
            created_at
        FROM predictions
        WHERE batch_id = %s
        ORDER BY created_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (batch_id,))
            return [dict(row) for row in cursor.fetchall()]
        

def get_batch_by_id(batch_id: str) -> dict[str, Any]:
    """Возвращает batch metadata по batch_id."""
    query = """
        SELECT
            batch_id,
            filename,
            object_key,
            s3_uri,
            status,
            uploaded_at,
            updated_at
        FROM batches
        WHERE batch_id = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (batch_id,))
            row = cursor.fetchone()

    if row is None:
        raise ValueError(f"Batch not found: {batch_id}")

    return dict(row)


def update_batch_status(batch_id: str, status: str) -> dict[str, Any]:
    """Обновляет статус batch-файла."""
    query = """
        UPDATE batches
        SET
            status = %s,
            updated_at = NOW()
        WHERE batch_id = %s
        RETURNING
            batch_id,
            filename,
            object_key,
            s3_uri,
            status,
            uploaded_at,
            updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (status, batch_id))
            row = cursor.fetchone()

    if row is None:
        raise ValueError(f"Batch not found: {batch_id}")

    return dict(row)

def list_clusters_by_batch(batch_id: str) -> list[dict[str, Any]]:
    """Возвращает clusters для конкретного batch с источником каждого профиля."""
    query = """
        SELECT
            c.cluster_record_id,
            c.cluster_id,
            c.batch_id AS cluster_batch_id,
            c.profile_id,
            c.cluster_size,
            c.recommendation_mode,
            c.created_at,

            p.first_seen_batch_id AS profile_first_seen_batch_id,
            p.last_seen_batch_id AS profile_last_seen_batch_id,
            p.source_filename AS profile_source_filename
        FROM clusters c
        LEFT JOIN profiles p
            ON c.profile_id = p.profile_id
        WHERE c.batch_id = %s
        ORDER BY c.cluster_size DESC, c.cluster_id, c.profile_id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (batch_id,))
            return [dict(row) for row in cursor.fetchall()]
        
        