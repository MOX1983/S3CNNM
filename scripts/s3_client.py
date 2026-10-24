import boto3
from botocore.exceptions import ClientError

from load_yml import configClient


def get_create_s3_resource():
    try:
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=configClient.ACCESS_KEY,
            aws_secret_access_key=configClient.SECRET_ACCESS_KEY,
            endpoint_url=configClient.ENDPOINT_URL,
        )
    except ClientError as e:
        raise f"ERROR {e}"
    return s3


def get_create_s3_client() -> boto3.session.Session.client:
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=configClient.ACCESS_KEY,
            aws_secret_access_key=configClient.SECRET_ACCESS_KEY,
            endpoint_url=configClient.ENDPOINT_URL,
        )
    except ClientError as e:
        raise f"ERROR {e}"


def get_obj(bucket: str, key: str):
    client = get_create_s3_client()
    return client.get_object(Bucket=bucket, Key=key)


def get_bytes_body_obj(bucket: str, key: str) -> bytes:
    return get_obj(bucket, key)["Body"].read()
