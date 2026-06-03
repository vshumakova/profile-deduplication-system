import pandas as pd

from app.decisions.recommendations import add_recommendations
from app.models.predict import load_model, predict_duplicate_scores


def run_model_inference(features_df: pd.DataFrame) -> pd.DataFrame:
    """Выполняет inference модели и добавляет бизнес-рекомендации."""
    predictions_df = predict_duplicate_scores(features_df)
    predictions_df = add_recommendations(predictions_df)

    return predictions_df


def run_test_pipeline() -> pd.DataFrame:
    """Тестовый запуск pipeline-worker без Redis, PostgreSQL, MinIO и Spark."""
    _, feature_cols, _, _ = load_model()

    features_df = pd.DataFrame(
        [
            {col: 0 for col in feature_cols},
            {col: 1 for col in feature_cols},
        ]
    )

    features_df["profile1"] = ["test_profile_1", "test_profile_2"]
    features_df["profile2"] = ["test_profile_3", "test_profile_4"]

    predictions_df = run_model_inference(features_df)

    return predictions_df


if __name__ == "__main__":
    result = run_test_pipeline()

    print("Pipeline worker test completed successfully")
    print(
        result[
            ["profile1", "profile2", "match_score", "is_duplicate", "recommendation"]
        ]
    )