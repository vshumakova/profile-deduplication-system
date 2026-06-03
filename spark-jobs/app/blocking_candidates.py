import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def create_spark_session() -> SparkSession:
    """Создаёт локальную SparkSession для candidate generation."""
    return (
        SparkSession.builder
        .appName("FlocktoryBlockingCandidateGeneration")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def read_input(spark: SparkSession, input_path: str) -> DataFrame:
    """Читает CSV или Parquet."""
    suffix = Path(input_path).suffix.lower()

    if suffix == ".csv":
        return (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(input_path)
        )

    if suffix == ".parquet":
        return spark.read.parquet(input_path)

    raise ValueError(f"Unsupported input format: {suffix}")


def normalize_phone(column: F.Column) -> F.Column:
    """Оставляет в телефоне только цифры."""
    return F.regexp_replace(F.coalesce(column.cast("string"), F.lit("")), r"[^0-9]", "")


def email_domain(column: F.Column) -> F.Column:
    """Достаёт домен email."""
    return F.lower(F.split(F.coalesce(column.cast("string"), F.lit("")), "@").getItem(1))


def add_blocking_keys(df: DataFrame) -> DataFrame:
    """Создаёт blocking keys для профилей.

    Один профиль может попасть в несколько блоков.
    Широкие признаки вроде email_domain не используются отдельно,
    а только в комбинации с именем/фамилией.
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

    blocks = (
        phone_blocks
        .unionByName(first_name_email_blocks)
        .unionByName(last_name_email_blocks)
        .unionByName(first_name_birthday_blocks)
        .unionByName(first_name_sex_blocks)
        .dropDuplicates(["profile_id", "blocking_key", "blocking_type"])
    )

    return blocks

def build_candidate_pairs(blocks: DataFrame) -> DataFrame:
    """Строит candidate pairs через self-join внутри одинакового blocking_key."""
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

    return pairs


def write_output(candidate_pairs: DataFrame, output_path: str) -> None:
    """Сохраняет candidate pairs в CSV."""
    (
        candidate_pairs
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(output_path)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default="/data/spark_candidate_pairs")
    args = parser.parse_args()

    spark = create_spark_session()

    print("Reading input:", args.input_path, flush=True)
    df = read_input(spark, args.input_path)

    rows_count = df.count()
    unique_profiles_count = df.select("profile_id").distinct().count()

    print(f"Rows count: {rows_count}", flush=True)
    print(f"Unique profile_id count: {unique_profiles_count}", flush=True)

    blocks = add_blocking_keys(df)
    blocks_count = blocks.count()
    unique_blocks_count = blocks.select("blocking_key").distinct().count()

    print(f"Blocking rows count: {blocks_count}", flush=True)
    print(f"Unique blocking keys count: {unique_blocks_count}", flush=True)

    candidate_pairs = build_candidate_pairs(blocks)
    candidate_pairs_count = candidate_pairs.count()

    print(f"Candidate pairs count: {candidate_pairs_count}", flush=True)

    print("Sample candidate pairs:", flush=True)
    candidate_pairs.show(20, truncate=False)

    print("Writing output:", args.output_path, flush=True)
    write_output(candidate_pairs, args.output_path)

    print("Done", flush=True)

    spark.stop()


if __name__ == "__main__":
    main()