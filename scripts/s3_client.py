import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
from pathlib import Path


PATH_TO_ENV = Path(__file__).parent.parent / ".env"
load_dotenv(PATH_TO_ENV)

def get_create_s3_resource():
    try:
        s3 = boto3.resource('s3',
                          aws_access_key_id=os.getenv("ACCESS_KEY"),
                          aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
                          endpoint_url=os.getenv("ENDPOINT_URL")
                          )
    except ClientError as e:
        raise f"ERROR {e}"
    return s3


