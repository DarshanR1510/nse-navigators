import redis
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

main_redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)