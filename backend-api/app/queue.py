import os

import redis


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "batch_processing_queue")


def get_redis_client() -> redis.Redis:
    """Создаёт Redis client."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
    )


def enqueue_batch(batch_id: str) -> None:
    """Добавляет batch_id в очередь обработки."""
    client = get_redis_client()
    client.lpush(REDIS_QUEUE_NAME, batch_id)