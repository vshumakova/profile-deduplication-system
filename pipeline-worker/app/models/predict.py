from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).parent / "artifacts" / "matching_model_v1.joblib"


def load_model(model_path: Path = MODEL_PATH):
    """Загружает сохранённый артефакт модели."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    fillna_value = artifact.get("fillna_value", -1)
    threshold = artifact.get("threshold", 0.85)

    return model, feature_cols, fillna_value, threshold


def predict_duplicate_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """Считает вероятность дубля для candidate pairs."""

    model, feature_cols, fillna_value, threshold = load_model()

    missing_cols = [col for col in feature_cols if col not in features_df.columns]

    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = features_df[feature_cols].fillna(fillna_value)

    result = features_df.copy()
    result["match_score"] = model.predict_proba(X)[:, 1]
    result["is_duplicate"] = result["match_score"] >= threshold

    return result