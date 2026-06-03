import os
import json
from typing import Any
from uuid import UUID

import pandas as pd

import psycopg2
from psycopg2.extras import RealDictCursor


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "flocktory")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


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


def get_batch_by_id(batch_id: str | UUID) -> dict[str, Any]:
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
            cursor.execute(query, (str(batch_id),))
            row = cursor.fetchone()

    if row is None:
        raise ValueError(f"Batch not found: {batch_id}")

    return dict(row)


def update_batch_status(batch_id: str | UUID, status: str) -> dict[str, Any]:
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
            cursor.execute(query, (status, str(batch_id)))
            row = cursor.fetchone()

    if row is None:
        raise ValueError(f"Batch not found: {batch_id}")

    return dict(row)

def save_predictions(batch_id: str | UUID, predictions: list[dict[str, Any]]) -> int:
    """Сохраняет predictions для batch в PostgreSQL."""
    query = """
        INSERT INTO predictions (
            batch_id,
            profile1,
            profile2,
            match_score,
            is_duplicate,
            recommendation
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    rows = [
        (
            str(batch_id),
            prediction["profile1"],
            prediction["profile2"],
            prediction["match_score"],
            prediction["is_duplicate"],
            prediction["recommendation"],
        )
        for prediction in predictions
    ]

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, rows)

    return len(rows)


def save_clusters(
    batch_id: str | UUID,
    clusters: list[dict[str, Any]],
    recommendation_mode: str = "auto_merge",
) -> int:
    """Сохраняет clusters в PostgreSQL.

    Перед сохранением удаляет старые clusters для batch_id и recommendation_mode,
    чтобы повторный запуск batch-а не создавал дубли.
    """

    delete_query = """
        DELETE FROM clusters
        WHERE batch_id = %s
          AND recommendation_mode = %s;
    """

    insert_query = """
        INSERT INTO clusters (
            cluster_id,
            batch_id,
            profile_id,
            cluster_size,
            recommendation_mode
        )
        VALUES (%s, %s, %s, %s, %s);
    """

    rows = [
        (
            cluster["cluster_id"],
            str(batch_id),
            cluster["profile_id"],
            cluster["cluster_size"],
            cluster["recommendation_mode"],
        )
        for cluster in clusters
    ]

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                delete_query,
                (
                    str(batch_id),
                    recommendation_mode,
                ),
            )

            if rows:
                cursor.executemany(insert_query, rows)

    return len(rows)

def is_null_value(value) -> bool:
    """Безопасно проверяет пустые scalar-значения."""
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        return False

    except (TypeError, ValueError):
        return False


def serialize_profile_value(value) -> str | None:
    """Приводит значение профиля к строке для сохранения в PostgreSQL."""
    if is_null_value(value):
        return None

    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value)


def first_non_empty_value(values: pd.Series):
    """Возвращает первое непустое значение из Series."""
    for value in values:
        if not is_null_value(value):
            return value

    return None


def aggregate_profiles_for_storage(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует batch до одной строки на profile_id для master table."""
    if "profile_id" not in batch_df.columns:
        raise ValueError("Column 'profile_id' is required")

    columns_to_keep = [
        "profile_id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "birthday",
        "sex",
        "created_at",
        "non_processing_features",
        "realtime_features",
        "fs_features",
    ]

    working_df = batch_df.copy()

    for column in columns_to_keep:
        if column not in working_df.columns:
            working_df[column] = None

    working_df["profile_id"] = working_df["profile_id"].astype(str)

    aggregated_df = (
        working_df[columns_to_keep]
        .groupby("profile_id", as_index=False)
        .agg(first_non_empty_value)
    )

    return aggregated_df


def save_profiles(
    batch_id: str | UUID,
    batch_df: pd.DataFrame,
    source_filename: str | None = None,
) -> int:
    """Сохраняет профили batch-а в master table profiles.

    Если profile_id уже есть, обновляет last_seen_batch_id, last_seen_at
    и актуальные поля профиля.
    """
    profiles_df = aggregate_profiles_for_storage(batch_df)

    if profiles_df.empty:
        return 0

    query = """
        INSERT INTO profiles (
            profile_id,
            first_seen_batch_id,
            last_seen_batch_id,
            source_filename,
            first_name,
            last_name,
            email,
            phone,
            birthday,
            sex,
            created_at,
            non_processing_features,
            realtime_features,
            fs_features
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (profile_id)
        DO UPDATE SET
            last_seen_batch_id = EXCLUDED.last_seen_batch_id,
            last_seen_at = NOW(),
            source_filename = EXCLUDED.source_filename,
            first_name = COALESCE(EXCLUDED.first_name, profiles.first_name),
            last_name = COALESCE(EXCLUDED.last_name, profiles.last_name),
            email = COALESCE(EXCLUDED.email, profiles.email),
            phone = COALESCE(EXCLUDED.phone, profiles.phone),
            birthday = COALESCE(EXCLUDED.birthday, profiles.birthday),
            sex = COALESCE(EXCLUDED.sex, profiles.sex),
            created_at = COALESCE(EXCLUDED.created_at, profiles.created_at),
            non_processing_features = COALESCE(EXCLUDED.non_processing_features, profiles.non_processing_features),
            realtime_features = COALESCE(EXCLUDED.realtime_features, profiles.realtime_features),
            fs_features = COALESCE(EXCLUDED.fs_features, profiles.fs_features);
    """

    rows = []

    for _, row in profiles_df.iterrows():
        rows.append(
            (
                serialize_profile_value(row["profile_id"]),
                str(batch_id),
                str(batch_id),
                source_filename,
                serialize_profile_value(row.get("first_name")),
                serialize_profile_value(row.get("last_name")),
                serialize_profile_value(row.get("email")),
                serialize_profile_value(row.get("phone")),
                serialize_profile_value(row.get("birthday")),
                serialize_profile_value(row.get("sex")),
                serialize_profile_value(row.get("created_at")),
                serialize_profile_value(row.get("non_processing_features")),
                serialize_profile_value(row.get("realtime_features")),
                serialize_profile_value(row.get("fs_features")),
            )
        )

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, rows)

    return len(rows)

def load_historical_profiles(exclude_batch_id: str | UUID) -> pd.DataFrame:
    """Загружает ранее сохранённые профили из master table profiles.

    Исключаем профили из текущего batch, чтобы не дублировать
    текущие строки при сравнении new ↔ historical.
    """
    query = """
        SELECT
            profile_id,
            first_name,
            last_name,
            email,
            phone,
            birthday,
            sex,
            created_at,
            non_processing_features,
            realtime_features,
            fs_features,
            first_seen_batch_id,
            last_seen_batch_id,
            source_filename
        FROM profiles
        WHERE last_seen_batch_id != %s;
    """

    with get_connection() as conn:
        historical_df = pd.read_sql_query(
            query,
            conn,
            params=(str(exclude_batch_id),),
        )

    return historical_df

