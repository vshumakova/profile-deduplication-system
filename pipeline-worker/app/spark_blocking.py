from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


def create_spark_session() -> SparkSession:
    """Создаёт локальную SparkSession для candidate generation."""
    return (
        SparkSession.builder
        .appName("FlocktoryPipelineWorkerBlocking")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def normalize_phone(column: F.Column) -> F.Column:
    """Оставляет в телефоне только цифры."""
    return F.regexp_replace(
        F.coalesce(column.cast("string"), F.lit("")),
        r"[^0-9]",
        "",
    )


def email_domain(column: F.Column) -> F.Column:
    """Достаёт домен email."""
    return F.lower(
        F.split(
            F.coalesce(column.cast("string"), F.lit("")),
            "@",
        ).getItem(1)
    )


def add_blocking_keys(df: DataFrame) -> DataFrame:
    """Создаёт blocking keys для профилей.

    Один profile_id может попасть в несколько блоков.
    """
    base = (
        df
        .withColumn("profile_id", F.col("profile_id").cast("string"))
        .withColumn("first_name_norm", F.lower(F.trim(F.col("first_name").cast("string"))))
        .withColumn("last_name_norm", F.lower(F.trim(F.col("last_name").cast("string"))))
        .withColumn("sex_norm", F.lower(F.trim(F.col("sex").cast("string"))))
        .withColumn("birthday_norm", F.col("birthday").cast("string"))
        .withColumn("phone_norm", normalize_phone(F.col("phone")))
        .withColumn("email_domain", email_domain(F.col("email")))
    )

    phone_blocks = (
        base
        .filter(F.length("phone_norm") >= 7)
        .select(
            "profile_id",
            F.concat(F.lit("phone:"), F.col("phone_norm")).alias("blocking_key"),
            F.lit("phone").alias("blocking_type"),
        )
    )

    first_name_email_blocks = (
        base
        .filter(
            F.col("first_name_norm").isNotNull()
            & (F.length("first_name_norm") > 0)
            & F.col("email_domain").isNotNull()
            & (F.length("email_domain") > 0)
        )
        .select(
            "profile_id",
            F.concat(
                F.lit("first_name_email_domain:"),
                F.col("first_name_norm"),
                F.lit(":"),
                F.col("email_domain"),
            ).alias("blocking_key"),
            F.lit("first_name_email_domain").alias("blocking_type"),
        )
    )

    last_name_email_blocks = (
        base
        .filter(
            F.col("last_name_norm").isNotNull()
            & (F.length("last_name_norm") > 0)
            & F.col("email_domain").isNotNull()
            & (F.length("email_domain") > 0)
        )
        .select(
            "profile_id",
            F.concat(
                F.lit("last_name_email_domain:"),
                F.col("last_name_norm"),
                F.lit(":"),
                F.col("email_domain"),
            ).alias("blocking_key"),
            F.lit("last_name_email_domain").alias("blocking_type"),
        )
    )

    first_name_birthday_blocks = (
        base
        .filter(
            F.col("first_name_norm").isNotNull()
            & (F.length("first_name_norm") > 0)
            & F.col("birthday_norm").isNotNull()
            & (F.length("birthday_norm") > 0)
        )
        .select(
            "profile_id",
            F.concat(
                F.lit("first_name_birthday:"),
                F.col("first_name_norm"),
                F.lit(":"),
                F.col("birthday_norm"),
            ).alias("blocking_key"),
            F.lit("first_name_birthday").alias("blocking_type"),
        )
    )

    first_name_sex_blocks = (
        base
        .filter(
            F.col("first_name_norm").isNotNull()
            & (F.length("first_name_norm") > 0)
            & F.col("sex_norm").isNotNull()
            & (F.length("sex_norm") > 0)
        )
        .select(
            "profile_id",
            F.concat(
                F.lit("first_name_sex:"),
                F.col("first_name_norm"),
                F.lit(":"),
                F.col("sex_norm"),
            ).alias("blocking_key"),
            F.lit("first_name_sex").alias("blocking_type"),
        )
    )

    return (
        phone_blocks
        .unionByName(first_name_email_blocks)
        .unionByName(last_name_email_blocks)
        .unionByName(first_name_birthday_blocks)
        .unionByName(first_name_sex_blocks)
        .dropDuplicates(["profile_id", "blocking_key", "blocking_type"])
    )


def to_safe_string(value):
    """Приводит значение к безопасной строке для Spark."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def build_candidate_pairs_with_spark(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Строит candidate pairs через PySpark blocking.

    Args:
        batch_df: Исходный batch с профилями.

    Returns:
        DataFrame с колонками profile1, profile2, blocking_key, blocking_type.
    """
    output_columns = [
        "profile1",
        "profile2",
        "blocking_key",
        "blocking_type",
    ]

    if batch_df.empty:
        return pd.DataFrame(columns=output_columns)

    if "profile_id" not in batch_df.columns:
        raise ValueError("Column 'profile_id' is required")

    blocking_columns = [
        "profile_id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "birthday",
        "sex",
    ]

    spark_input = batch_df.copy()

    for column in blocking_columns:
        spark_input[column] = spark_input[column].apply(to_safe_string)

    schema = StructType(
            [
                StructField("profile_id", StringType(), True),
                StructField("first_name", StringType(), True),
                StructField("last_name", StringType(), True),
                StructField("email", StringType(), True),
                StructField("phone", StringType(), True),
                StructField("birthday", StringType(), True),
                StructField("sex", StringType(), True),
            ]
        )

    spark = create_spark_session()

    try:
        spark_df = spark.createDataFrame(
            spark_input.to_dict("records"),
            schema=schema,
        )

        blocks = add_blocking_keys(spark_df)

        left = blocks.alias("left")
        right = blocks.alias("right")

        pairs = (
            left
            .join(
                right,
                on=[
                    F.col("left.blocking_key") == F.col("right.blocking_key"),
                    F.col("left.profile_id") < F.col("right.profile_id"),
                ],
                how="inner",
            )
            .select(
                F.col("left.profile_id").alias("profile1"),
                F.col("right.profile_id").alias("profile2"),
                F.col("left.blocking_key").alias("blocking_key"),
                F.col("left.blocking_type").alias("blocking_type"),
            )
            .dropDuplicates(["profile1", "profile2"])
        )

        return pairs.toPandas()

    finally:
        spark.stop()