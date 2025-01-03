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

    redisClient = redis.Redis(host=redisURL, port=6379, db=0)

    if redisClient is None:
        print("Redis initialization failure!")
    else:
        print("Redis initialization success!")


"""
    Gets raw bytes from the key in redis
"""
def getFromRedis(key):
    print("Retrieving from redis...")
    assert(redisClient is not None)

    result = None
    try:
        result = redisClient.get(key)
    except Exception as e:
        print("Exception!")
        print(e)

    if result: 
        print("Retrieved from redis!")
    else:
        print("Not present in redis!")

    return result
