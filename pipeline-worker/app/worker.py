import os
import time

import redis

from app.run_batch import run_batch


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


def listen_queue() -> None:
    """Бесконечно слушает Redis queue и запускает batch processing."""
    client = get_redis_client()

    print("Pipeline worker started")
    print(f"Listening Redis queue: {REDIS_QUEUE_NAME}")

    while True:
        try:
            _, batch_id = client.brpop(REDIS_QUEUE_NAME)
            print(f"Received batch_id from queue: {batch_id}")

            run_batch(batch_id)

        except Exception as error:
            print(f"Worker error: {error}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    listen_queue()