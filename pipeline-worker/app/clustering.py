from __future__ import annotations

from collections import defaultdict, deque
from uuid import NAMESPACE_URL, uuid5

import pandas as pd


def build_profile_clusters(
    predictions_df: pd.DataFrame,
    batch_id: str,
    recommendation_mode: str = "auto_merge",
) -> list[dict]:
    """Строит кластеры профилей на основе pairwise predictions.

    Args:
        predictions_df: DataFrame с колонками profile1, profile2, recommendation.
        batch_id: UUID batch-а.
        recommendation_mode: Режим кластеризации.
            Сейчас основной production-safe режим — auto_merge.

    Returns:
        Список строк для сохранения в PostgreSQL:
        cluster_id, batch_id, profile_id, cluster_size, recommendation_mode.
    """
    required_columns = {"profile1", "profile2", "recommendation"}
    missing_columns = required_columns - set(predictions_df.columns)

    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    if predictions_df.empty:
        return []

    if recommendation_mode == "auto_merge":
        cluster_edges = predictions_df[
            predictions_df["recommendation"] == "auto_merge"
        ].copy()
    elif recommendation_mode == "auto_merge_or_manual_review":
        cluster_edges = predictions_df[
            predictions_df["recommendation"].isin(
                ["auto_merge", "manual_review"]
            )
        ].copy()
    else:
        raise ValueError(f"Unsupported recommendation_mode: {recommendation_mode}")

    if cluster_edges.empty:
        return []

    graph = defaultdict(set)

    for _, row in cluster_edges.iterrows():
        profile1 = str(row["profile1"])
        profile2 = str(row["profile2"])

        if not profile1 or not profile2 or profile1 == profile2:
            continue

        graph[profile1].add(profile2)
        graph[profile2].add(profile1)

    visited = set()
    cluster_rows = []

    for start_profile in sorted(graph.keys()):
        if start_profile in visited:
            continue

        queue = deque([start_profile])
        component = []

        visited.add(start_profile)

        while queue:
            current_profile = queue.popleft()
            component.append(current_profile)

            for neighbor in sorted(graph[current_profile]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) < 2:
            continue

        component = sorted(component)

        cluster_id = uuid5(
            NAMESPACE_URL,
            f"{batch_id}:{recommendation_mode}:{'|'.join(component)}",
        )

        for profile_id in component:
            cluster_rows.append(
                {
                    "cluster_id": str(cluster_id),
                    "batch_id": str(batch_id),
                    "profile_id": profile_id,
                    "cluster_size": len(component),
                    "recommendation_mode": recommendation_mode,
                }
            )

    return cluster_rows