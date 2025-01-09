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

    aws_region = "us-east-2"

    aws_access_key_id = os.getenv("S3_ACCESS_KEY", "")
    aws_access_key_secret = os.getenv("S3_ACCESS_SECRET","")
    
    try:
        s3_client = None
        if os.getenv("ENV") == "DEV":
            s3_client = boto3.client(
                "s3", 
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_access_key_secret,
            )
        else:
            s3_client = boto3.client(
                "s3", 
                region_name=aws_region,
            )

        s3_client.head_bucket(Bucket=bucket_name)
        print("S3 initialization success!")
    except (BotoCoreError, ClientError) as e:
        print(f"S3 Initialization failure: {e}")
        raise

def getFromS3(key, bucket):
    """
    Retrieves raw bytes from S3 for the given key.
    """

    print("Getting from S3...")
    assert s3_client is not None, "S3 client not initialized"
    assert bucket_name is not None, "Bucket name not set"

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        print(f"Retrieved from S3 with key: {key}")
        return response["Body"].read()
    except (BotoCoreError, ClientError) as e:
        print(f"Failed to retrieve from S3: {e}")
        return None


aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-2.amazonaws.com

763104351884.dkr.ecr.us-east-2.amazonaws.com
