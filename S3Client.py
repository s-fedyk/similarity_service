import boto3
import os
from botocore.exceptions import BotoCoreError, ClientError
from botocore import UNSIGNED
from botocore.client import Config

s3_client = None
bucket_name = None

def initS3():
    """
    Initializes the S3 client and checks the bucket availability.
    """
    global s3_client, bucket_name
    print("Initializing S3...")

    bucket_name = os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("S3_BUCKET environment variable is not set")

    aws_region = os.getenv("AWS_REGION", "us-east-2")

    aws_access_key_id = os.getenv("S3_ACCESS_KEY", "")
    aws_access_key_secret = os.getenv("S3_ACCESS_SECRET","")
    
    try:
        config = Config()
        if (os.getenv("ENV") != "DEV"):
            # don't sign inside vpc
            config=Config(signature_version=UNSIGNED)

        s3_client = boto3.client(
            "s3", 
            region_name=aws_region,
            config=config,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_access_key_secret,
        )
        # Check if the bucket exists and is accessible
        s3_client.head_bucket(Bucket=bucket_name)
        print("S3 initialization success!")
    except (BotoCoreError, ClientError) as e:
        print(f"S3 Initialization failure: {e}")
        raise

def getFromS3(key):
    """
    Retrieves raw bytes from S3 for the given key.
    """
    assert s3_client is not None, "S3 client not initialized"
    assert bucket_name is not None, "Bucket name not set"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        print(f"Retrieved from S3 with key: {key}")
        return response["Body"].read()
    except (BotoCoreError, ClientError) as e:
        print(f"Failed to retrieve from S3: {e}")
        return None
