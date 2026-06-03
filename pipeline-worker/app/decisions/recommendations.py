import pandas as pd


def add_recommendations(
    predictions_df: pd.DataFrame,
    auto_merge_threshold: float = 0.90,
    manual_review_threshold: float = 0.65,
) -> pd.DataFrame:
    result = predictions_df.copy()

    def decide(score: float) -> str:
        if score >= auto_merge_threshold:
            return "auto_merge"
        if score >= manual_review_threshold:
            return "manual_review"
        return "no_duplicate"

    result["recommendation"] = result["match_score"].apply(decide)

    return result