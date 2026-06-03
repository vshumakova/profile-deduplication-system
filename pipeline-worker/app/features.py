import ast
import json
import re
from itertools import combinations
from typing import Any

import pandas as pd


MODEL_FEATURE_COLUMNS = [
    "time_diff_hours",
    "event_count_diff",
    "fs_jaccard",
    "fs_count_diff",
    "same_geoid",
    "same_first_name",
    "same_phone",
    "same_sex",
    "same_email_domain",
    "same_np_device",
    "local_hour_mean_diff",
]


def normalize_text(value: object) -> str | None:
    """Нормализует текстовое значение для сравнения."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"", "nan", "none", "null"}:
        return None

    return value


def safe_equal(left: object, right: object) -> int:
    """Безопасно сравнивает два значения после нормализации."""
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)

    if left_norm is None or right_norm is None:
        return 0

    return int(left_norm == right_norm)


def get_email_domain(email: object) -> str | None:
    """Возвращает домен email."""
    email_norm = normalize_text(email)

    if email_norm is None or "@" not in email_norm:
        return None

    return email_norm.split("@")[-1]


def is_null_scalar(value) -> bool:
    """Безопасно проверяет, является ли значение пустым scalar-значением."""
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        # Если pd.isna вернул массив, значит это не scalar.
        return False

    except (TypeError, ValueError):
        return False


def parse_feature_list(value) -> list[str]:
    """Парсит feature-list из parquet/csv в список строк.

    Поддерживает:
    - None / NaN
    - list / tuple / set
    - numpy array
    - строку вида "['a:1' 'b:2']"
    - строку вида "['a:1', 'b:2']"
    - dict
    """
    if is_null_scalar(value):
        return []

    # Parquet может вернуть numpy array.
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()

    if isinstance(value, dict):
        return [
            f"{key}:{item_value}"
            for key, item_value in value.items()
            if not is_null_scalar(item_value)
        ]

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if not is_null_scalar(item) and str(item).strip()
        ]

    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []

    raw_value = value.strip()

    if not raw_value:
        return []

    # Пробуем распарсить нормальный Python-list/dict.
    try:
        parsed = ast.literal_eval(raw_value)

        if isinstance(parsed, dict):
            return [
                f"{key}:{item_value}"
                for key, item_value in parsed.items()
                if not is_null_scalar(item_value)
            ]

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(item).strip()
                for item in parsed
                if not is_null_scalar(item) and str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    # Для строк вида: ['a:1' 'b:2'] без запятых.
    quoted_tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw_value)

    if quoted_tokens:
        return [
            first or second
            for first, second in quoted_tokens
            if (first or second).strip()
        ]

    # Fallback: убираем скобки и делим по пробелам.
    cleaned = (
        raw_value
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " ")
        .strip()
    )

    return [
        token.strip().strip("'").strip('"')
        for token in cleaned.split()
        if token.strip()
    ]


def parse_realtime_features(value: object) -> dict[str, Any]:
    """Парсит realtime_features как JSON."""
    if pd.isna(value):
        return {}

    if isinstance(value, dict):
        return value

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed


def get_feature_value(features: set[str], prefix: str) -> str | None:
    """Достаёт значение из set токенов формата prefix:value."""
    prefix = prefix.lower() + ":"

    for feature in features:
        if feature.startswith(prefix):
            return feature.split(":", 1)[1]

    return None


def jaccard_similarity(left, right) -> float:
    """Считает Jaccard similarity между двумя списками признаков.

    Args:
        left: Первый набор признаков.
        right: Второй набор признаков.

    Returns:
        Значение Jaccard similarity от 0 до 1.
    """
    left_set = set(left or [])
    right_set = set(right or [])

    if not left_set and not right_set:
        return 0.0

    intersection = left_set & right_set
    union = left_set | right_set

    if not union:
        return 0.0

    return len(intersection) / len(union)


def time_diff_hours(left_created_at: object, right_created_at: object) -> float:
    """Считает разницу между created_at в часах."""
    left_dt = pd.to_datetime(left_created_at, errors="coerce", utc=True)
    right_dt = pd.to_datetime(right_created_at, errors="coerce", utc=True)

    if pd.isna(left_dt) or pd.isna(right_dt):
        return 0.0

    return abs((left_dt - right_dt).total_seconds()) / 3600


def local_hour(value: object) -> float | None:
    """Возвращает час из created_at."""
    dt = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(dt):
        return None

    return float(dt.hour)


def local_hour_mean_diff(left_created_at: object, right_created_at: object) -> float:
    """Считает разницу часа события.

    В MVP используем created_at. Позже можно заменить на агрегат из realtime_features.
    """
    left_hour = local_hour(left_created_at)
    right_hour = local_hour(right_created_at)

    if left_hour is None or right_hour is None:
        return 0.0

    return abs(left_hour - right_hour)


def build_candidate_pairs(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Строит пары кандидатов внутри batch.

    MVP-версия: полный перебор разных profile_id.
    Если один profile_id встречается в нескольких строках, self-pairs исключаются.
    Позже заменим на blocking в PySpark.
    """
    if "profile_id" not in batch_df.columns:
        raise ValueError("Column 'profile_id' is required")

    rows = []
    seen_pairs = set()

    for left_idx, right_idx in combinations(batch_df.index, 2):
        left = batch_df.loc[left_idx]
        right = batch_df.loc[right_idx]

        left_profile_id = str(left["profile_id"])
        right_profile_id = str(right["profile_id"])

        # Не сравниваем профиль сам с собой
        if left_profile_id == right_profile_id:
            continue

        # Нормализуем порядок пары, чтобы A-B и B-A не считались разными
        pair_key = tuple(sorted([left_profile_id, right_profile_id]))

        # Не сохраняем одну и ту же пару profile_id повторно
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        rows.append(
            {
                "profile1": pair_key[0],
                "profile2": pair_key[1],
                "left_index": left_idx,
                "right_index": right_idx,
            }
        )

    return pd.DataFrame(rows)


def build_features(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Строит features_df с колонками, которые ожидает LightGBM-модель."""
    candidate_pairs = build_candidate_pairs(batch_df)

    if candidate_pairs.empty:
        return pd.DataFrame(columns=["profile1", "profile2", *MODEL_FEATURE_COLUMNS])

    feature_rows = []

    for _, pair in candidate_pairs.iterrows():
        left = batch_df.loc[pair["left_index"]]
        right = batch_df.loc[pair["right_index"]]

        left_email_domain = get_email_domain(left.get("email"))
        right_email_domain = get_email_domain(right.get("email"))

        left_np_features = parse_feature_list(left.get("non_processing_features"))
        right_np_features = parse_feature_list(right.get("non_processing_features"))

        left_fs_features = parse_feature_list(left.get("fs_features"))
        right_fs_features = parse_feature_list(right.get("fs_features"))

        left_geoid = get_feature_value(left_np_features, "geoname_id")
        right_geoid = get_feature_value(right_np_features, "geoname_id")

        left_device = get_feature_value(left_np_features, "device")
        right_device = get_feature_value(right_np_features, "device")

        feature_rows.append(
            {
                "profile1": pair["profile1"],
                "profile2": pair["profile2"],

                "time_diff_hours": time_diff_hours(
                    left.get("created_at"),
                    right.get("created_at"),
                ),
                "event_count_diff": 0,
                "fs_jaccard": jaccard_similarity(
                    left_fs_features,
                    right_fs_features,
                ),
                "fs_count_diff": abs(
                    len(left_fs_features) - len(right_fs_features)
                ),
                "same_geoid": int(
                    left_geoid is not None
                    and right_geoid is not None
                    and left_geoid == right_geoid
                ),
                "same_first_name": safe_equal(
                    left.get("first_name"),
                    right.get("first_name"),
                ),
                "same_phone": safe_equal(
                    left.get("phone"),
                    right.get("phone"),
                ),
                "same_sex": safe_equal(
                    left.get("sex"),
                    right.get("sex"),
                ),
                "same_email_domain": int(
                    left_email_domain is not None
                    and right_email_domain is not None
                    and left_email_domain == right_email_domain
                ),
                "same_np_device": int(
                    left_device is not None
                    and right_device is not None
                    and left_device == right_device
                ),
                "local_hour_mean_diff": local_hour_mean_diff(
                    left.get("created_at"),
                    right.get("created_at"),
                ),
            }
        )

    return pd.DataFrame(feature_rows)


def build_features_for_candidate_pairs(
    batch_df: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Строит features_df для заранее подготовленных candidate pairs.

    Args:
        batch_df: Исходные профили batch-а.
        candidate_pairs: DataFrame с колонками profile1/profile2.

    Returns:
        DataFrame с profile1, profile2 и признаками модели.
    """
    required_columns = {"profile1", "profile2"}
    missing_columns = required_columns - set(candidate_pairs.columns)

    if missing_columns:
        raise ValueError(f"Missing candidate pair columns: {missing_columns}")

    if candidate_pairs.empty:
        return pd.DataFrame(columns=["profile1", "profile2", *MODEL_FEATURE_COLUMNS])

    batch_df = batch_df.copy()
    batch_df["profile_id"] = batch_df["profile_id"].astype(str)

    # Для MVP берём первую строку на profile_id.
    # Данные не удаляются из batch-а, но для одной пары нужен один набор признаков.
    profiles_by_id = (
        batch_df
        .drop_duplicates(subset=["profile_id"], keep="first")
        .set_index("profile_id")
    )

    feature_rows = []

    for _, pair in candidate_pairs.iterrows():
        profile1 = str(pair["profile1"])
        profile2 = str(pair["profile2"])

        if profile1 == profile2:
            continue

        if profile1 not in profiles_by_id.index or profile2 not in profiles_by_id.index:
            continue

        left = profiles_by_id.loc[profile1]
        right = profiles_by_id.loc[profile2]

        left_email_domain = get_email_domain(left.get("email"))
        right_email_domain = get_email_domain(right.get("email"))

        left_np_features = parse_feature_list(left.get("non_processing_features"))
        right_np_features = parse_feature_list(right.get("non_processing_features"))

        left_fs_features = parse_feature_list(left.get("fs_features"))
        right_fs_features = parse_feature_list(right.get("fs_features"))

        left_geoid = get_feature_value(left_np_features, "geoname_id")
        right_geoid = get_feature_value(right_np_features, "geoname_id")

        left_device = get_feature_value(left_np_features, "device")
        right_device = get_feature_value(right_np_features, "device")

        feature_rows.append(
            {
                "profile1": profile1,
                "profile2": profile2,
                "time_diff_hours": time_diff_hours(
                    left.get("created_at"),
                    right.get("created_at"),
                ),
                "event_count_diff": 0,
                "fs_jaccard": jaccard_similarity(
                    left_fs_features,
                    right_fs_features,
                ),
                "fs_count_diff": abs(
                    len(left_fs_features) - len(right_fs_features)
                ),
                "same_geoid": int(
                    left_geoid is not None
                    and right_geoid is not None
                    and left_geoid == right_geoid
                ),
                "same_first_name": safe_equal(
                    left.get("first_name"),
                    right.get("first_name"),
                ),
                "same_phone": safe_equal(
                    left.get("phone"),
                    right.get("phone"),
                ),
                "same_sex": safe_equal(
                    left.get("sex"),
                    right.get("sex"),
                ),
                "same_email_domain": int(
                    left_email_domain is not None
                    and right_email_domain is not None
                    and left_email_domain == right_email_domain
                ),
                "same_np_device": int(
                    left_device is not None
                    and right_device is not None
                    and left_device == right_device
                ),
                "local_hour_mean_diff": local_hour_mean_diff(
                    left.get("created_at"),
                    right.get("created_at"),
                ),
            }
        )

    return pd.DataFrame(feature_rows)

