import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Flocktory Profile Deduplication",
    page_icon="🧩",
    layout="wide",
)


def api_get(path: str) -> dict:
    response = requests.get(
        f"{BACKEND_API_URL}{path}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, files=None) -> dict:
    response = requests.post(
        f"{BACKEND_API_URL}{path}",
        files=files,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def format_datetime(value) -> str:
    if not value:
        return ""

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except Exception:
        return str(value)


def load_batches() -> pd.DataFrame:
    data = api_get("/batches")
    batches = data.get("batches", [])

    if not batches:
        return pd.DataFrame()

    df = pd.DataFrame(batches)

    for column in ["uploaded_at", "updated_at"]:
        if column in df.columns:
            df[column] = df[column].apply(format_datetime)

    return df


def load_predictions(batch_id: str) -> pd.DataFrame:
    data = api_get(f"/batches/{batch_id}/predictions")
    predictions = data.get("predictions", [])

    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame(predictions)

    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(format_datetime)

    if "match_score" in df.columns:
        df["match_score"] = df["match_score"].astype(float)

    return df


def load_clusters(batch_id: str) -> pd.DataFrame:
    data = api_get(f"/batches/{batch_id}/clusters")
    clusters = data.get("clusters", [])

    if not clusters:
        return pd.DataFrame()

    df = pd.DataFrame(clusters)

    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(format_datetime)

    return df


def render_status_badge(status: str) -> str:
    status = str(status)

    emoji_by_status = {
        "uploaded": "📥",
        "queued": "⏳",
        "processing": "⚙️",
        "completed": "✅",
        "failed": "❌",
    }

    return f"{emoji_by_status.get(status, 'ℹ️')} {status}"


def render_header() -> None:
    st.title("🧩 Flocktory Profile Deduplication")
    st.caption(
        "MVP-сервис для поиска дублей клиентских профилей: "
        "MinIO + PostgreSQL + Redis + PySpark Blocking + LightGBM."
    )


def render_upload_tab() -> None:
    st.subheader("📤 Загрузка batch-файла")

    st.info(
        "Поддерживаются CSV и Parquet файлы. "
        "Файл сохраняется в MinIO, а metadata и статус — в PostgreSQL."
    )

    uploaded_file = st.file_uploader(
        "Выберите batch-файл",
        type=["csv", "parquet"],
    )

    if uploaded_file is None:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Имя файла", uploaded_file.name)

    with col2:
        st.metric("Размер", f"{uploaded_file.size / 1024:.1f} KB")

    with col3:
        extension = uploaded_file.name.split(".")[-1].upper()
        st.metric("Формат", extension)

    if st.button("Загрузить batch", type="primary"):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }

            result = api_post("/upload", files=files)
            batch = result["batch"]

            st.success("Batch успешно загружен")
            st.code(str(batch["batch_id"]), language="text")

        except requests.HTTPError as error:
            st.error(f"Ошибка API: {error.response.text}")
        except Exception as error:
            st.error(f"Ошибка загрузки: {error}")


def render_batches_tab() -> None:
    st.subheader("📦 Batch-файлы")

    try:
        batches_df = load_batches()
    except Exception as error:
        st.error(f"Не удалось загрузить batches: {error}")
        return

    if batches_df.empty:
        st.warning("Пока нет загруженных batch-файлов.")
        return

    status_counts = batches_df["status"].value_counts().to_dict()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Всего batches", len(batches_df))

    with col2:
        st.metric("Completed", status_counts.get("completed", 0))

    with col3:
        st.metric("Processing / queued", status_counts.get("processing", 0) + status_counts.get("queued", 0))

    with col4:
        st.metric("Failed", status_counts.get("failed", 0))

    display_df = batches_df.copy()

    if "status" in display_df.columns:
        display_df["status"] = display_df["status"].apply(render_status_badge)

    st.dataframe(
        display_df[
            [
                "batch_id",
                "filename",
                "status",
                "uploaded_at",
                "updated_at",
                "object_key",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("▶️ Запуск обработки")

    batch_options = {
        f"{row['filename']} | {row['batch_id']} | {row['status']}": row["batch_id"]
        for _, row in batches_df.iterrows()
    }

    selected_label = st.selectbox(
        "Выберите batch",
        options=list(batch_options.keys()),
    )

    selected_batch_id = batch_options[selected_label]

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Запустить обработку", type="primary"):
            try:
                result = api_post(f"/batches/{selected_batch_id}/process")
                st.success(result.get("message", "Batch поставлен в очередь"))
                st.json(result)
            except requests.HTTPError as error:
                st.error(f"Ошибка API: {error.response.text}")
            except Exception as error:
                st.error(f"Ошибка запуска: {error}")

    with col2:
        st.caption(
            "После запуска Backend API кладёт batch_id в Redis Queue, "
            "а Pipeline Worker забирает задачу и запускает ML-пайплайн."
        )


def render_predictions_tab() -> None:
    st.subheader("🔎 Pairwise predictions")

    try:
        batches_df = load_batches()
    except Exception as error:
        st.error(f"Не удалось загрузить batches: {error}")
        return

    if batches_df.empty:
        st.warning("Сначала загрузите batch-файл.")
        return

    batch_options = {
        f"{row['filename']} | {row['batch_id']} | {row['status']}": row["batch_id"]
        for _, row in batches_df.iterrows()
    }

    selected_label = st.selectbox(
        "Выберите batch для просмотра predictions",
        options=list(batch_options.keys()),
        key="predictions_batch_select",
    )

    selected_batch_id = batch_options[selected_label]

    try:
        predictions_df = load_predictions(selected_batch_id)
    except Exception as error:
        st.error(f"Не удалось загрузить predictions: {error}")
        return

    if predictions_df.empty:
        st.warning("Для этого batch пока нет predictions.")
        return

    total_predictions = len(predictions_df)
    auto_merge_count = int((predictions_df["recommendation"] == "auto_merge").sum())
    manual_review_count = int((predictions_df["recommendation"] == "manual_review").sum())
    no_duplicate_count = int((predictions_df["recommendation"] == "no_duplicate").sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Всего пар", total_predictions)

    with col2:
        st.metric("Auto merge", auto_merge_count)

    with col3:
        st.metric("Manual review", manual_review_count)

    with col4:
        st.metric("No duplicate", no_duplicate_count)

    st.divider()

    recommendation_filter = st.multiselect(
        "Фильтр по рекомендации",
        options=sorted(predictions_df["recommendation"].unique()),
        default=sorted(predictions_df["recommendation"].unique()),
    )

    min_score = st.slider(
        "Минимальный match_score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )

    filtered_df = predictions_df[
        predictions_df["recommendation"].isin(recommendation_filter)
        & (predictions_df["match_score"] >= min_score)
    ].copy()

    filtered_df = filtered_df.sort_values("match_score", ascending=False)

    st.dataframe(
        filtered_df[
            [
                "profile1",
                "profile2",
                "match_score",
                "is_duplicate",
                "recommendation",
                "created_at",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_clusters_tab() -> None:
    st.subheader("🕸️ Кластеры профилей")

    st.info(
        "Кластеры строятся поверх pairwise predictions. "
        "Пары с рекомендацией auto_merge превращаются в граф связей, "
        "а connected components формируют группы профилей одного пользователя."
    )

    try:
        batches_df = load_batches()
    except Exception as error:
        st.error(f"Не удалось загрузить batches: {error}")
        return

    if batches_df.empty:
        st.warning("Сначала загрузите batch-файл.")
        return

    batch_options = {
        f"{row['filename']} | {row['batch_id']} | {row['status']}": row["batch_id"]
        for _, row in batches_df.iterrows()
    }

    selected_label = st.selectbox(
        "Выберите batch для просмотра clusters",
        options=list(batch_options.keys()),
        key="clusters_batch_select",
    )

    selected_batch_id = batch_options[selected_label]

    try:
        clusters_df = load_clusters(selected_batch_id)
    except Exception as error:
        st.error(f"Не удалось загрузить clusters: {error}")
        return

    if clusters_df.empty:
        st.warning("Для этого batch пока нет кластеров.")
        return

    unique_clusters = clusters_df["cluster_id"].nunique()
    profiles_in_clusters = clusters_df["profile_id"].nunique()
    max_cluster_size = int(clusters_df["cluster_size"].max())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Кластеров", unique_clusters)

    with col2:
        st.metric("Профилей в кластерах", profiles_in_clusters)

    with col3:
        st.metric("Макс. размер кластера", max_cluster_size)

    st.divider()

    st.subheader("Сводка по кластерам")

    cluster_summary = (
        clusters_df.groupby(["cluster_id", "cluster_size", "recommendation_mode"])
        .agg(
            profiles_count=("profile_id", "count"),
            created_at=("created_at", "min"),
        )
        .reset_index()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
    )

    for _, cluster in cluster_summary.iterrows():
        cluster_id = cluster["cluster_id"]
        cluster_size = int(cluster["cluster_size"])
        recommendation_mode = cluster["recommendation_mode"]

        cluster_profiles_df = (
            clusters_df[clusters_df["cluster_id"] == cluster_id]
            .sort_values("profile_id")
            .reset_index(drop=True)
        )

        title = (
            f"Кластер {cluster_id} | "
            f"профилей: {cluster_size} | "
            f"режим: {recommendation_mode}"
        )

        with st.expander(title, expanded=False):
            st.markdown("**Профили в кластере:**")

            display_columns = [
                "profile_id",
                "profile_first_seen_batch_id",
                "profile_last_seen_batch_id",
                "profile_source_filename",
                "cluster_size",
                "recommendation_mode",
                "created_at",
            ]

            existing_display_columns = [
                column for column in display_columns
                if column in cluster_profiles_df.columns
            ]

            profiles_display_df = cluster_profiles_df[existing_display_columns].copy()

            profiles_display_df = profiles_display_df.rename(
                columns={
                    "profile_id": "profile_id",
                    "cluster_size": "размер кластера",
                    "recommendation_mode": "режим",
                    "created_at": "дата создания",
                }
            )

            st.dataframe(
                profiles_display_df,
                use_container_width=True,
                hide_index=True,
            )


def render_upload_tab() -> None:
    """Рендерит вкладку загрузки batch-файла."""
    st.header("📤 Загрузка batch-файла")

    st.markdown(
        """
        Загрузите CSV или Parquet файл с профилями пользователей.
        
        После загрузки файл будет сохранён в MinIO, а metadata batch-а — в PostgreSQL.
        """
    )

    uploaded_file = st.file_uploader(
        label="Перетащите файл сюда или выберите его вручную",
        type=["csv", "parquet"],
        accept_multiple_files=False,
        help="Поддерживаются форматы .csv и .parquet",
    )

    if uploaded_file is None:
        st.info("Ожидаю файл для загрузки.")
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.success("Файл выбран")

    st.write("**Имя файла:**", uploaded_file.name)
    st.write("**Размер:**", f"{file_size_mb:.2f} MB")
    st.write("**Тип:**", uploaded_file.type or "unknown")

    if st.button("Загрузить batch в систему", type="primary"):
        with st.spinner("Загружаю файл в Backend API..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }

            try:
                response = requests.post(
                    f"{BACKEND_API_URL}/upload",
                    files=files,
                    timeout=60,
                )

                if response.status_code != 200:
                    st.error(f"Ошибка загрузки: {response.status_code}")
                    st.code(response.text)
                    return

                result = response.json()
                batch = result.get("batch", {})

                st.success("Batch успешно загружен")

                st.json(
                    {
                        "batch_id": batch.get("batch_id"),
                        "filename": batch.get("filename"),
                        "status": batch.get("status"),
                        "s3_uri": batch.get("s3_uri"),
                    }
                )

            except requests.RequestException as error:
                st.error("Не удалось отправить файл в Backend API")
                st.code(str(error))


def render_about_tab() -> None:
    """Рендерит вкладку с описанием пайплайна."""
    st.header("ℹ️ О пайплайне")

    st.markdown(
        """
        MVP реализует end-to-end pipeline для поиска и кластеризации дублей профилей.

        **Основной сценарий:**

        1. Пользователь загружает batch-файл через Streamlit.
        2. Backend API сохраняет файл в MinIO и metadata batch-а в PostgreSQL.
        3. При запуске обработки Backend API кладёт `batch_id` в Redis Queue.
        4. Pipeline Worker забирает задачу из очереди.
        5. Worker скачивает batch-файл из MinIO.
        6. Профили сохраняются в master table `profiles`.
        7. Worker загружает historical profiles из PostgreSQL.
        8. PySpark Blocking строит пары-кандидаты.
        9. pandas Feature Engineering считает признаки для пар.
        10. LightGBM выдаёт `match_score`.
        11. Business rules формируют рекомендацию:
            - `auto_merge`
            - `manual_review`
            - `no_duplicate`
        12. Система строит кластеры профилей и сохраняет результаты в PostgreSQL.

        **Важно:**  
        `entity_id` используется только для offline evaluation и не подаётся в модель как признак.
        """
    )


def main() -> None:
    render_header()

    tabs = st.tabs(
        [
            "📤 Загрузка",
            "📦 Батчи",
            "🔎 Предсказания",
            "🕸️ Кластеры",
            "ℹ️ О пайплайне",
        ]
    )

    with tabs[0]:
        render_upload_tab()

    with tabs[1]:
        render_batches_tab()

    with tabs[2]:
        render_predictions_tab()

    with tabs[3]:
        render_clusters_tab()

    with tabs[4]:
        render_about_tab()


if __name__ == "__main__":
    main()