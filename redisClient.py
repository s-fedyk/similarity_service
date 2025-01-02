import redis
import os

redisClient = None

def initRedis():
    print("Initializing redis...")
    global redisClient
    redisURL = os.getenv("REDIS_URL")

    if not redisURL:
        print("REDIS_URL not set")

    assert(redisURL)

    redisClient = redis.Redis(host=redisURL, port=6379, db=0, decode_responses=True)

    if redisClient is None:
        print("Redis initialization failure!")
    else:
        print("Redis initialization success!")


def getFromRedis(key):
    assert(redisClient is not None)

    result = redisClient.get(key)

    if result: 
        print("Retrieved from redis!")
    else:
        print("Not present in redis!")

    return result
