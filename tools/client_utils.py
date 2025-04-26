import os
from typing import Any
import boto3

# Create and return a Bedrock client
def get_bedrock_client(service: str = 'bedrock') -> Any:
    return boto3.client(service, region_name=os.getenv('AWS_REGION', 'us-west-2'))

# Create and return a Bedrock client
def get_bedrock_runtime_client(service: str = 'bedrock-runtime') -> Any:
    return boto3.client(service, region_name=os.getenv('AWS_REGION', 'us-west-2'))

# Create and return a Polly client
def get_polly_client(service: str = 'polly') -> Any:
    return boto3.client(service, region_name=os.getenv('AWS_REGION', 'us-west-2'))

# Create and return a S3 client
def get_s3_client(service: str = 's3') -> Any:
    return boto3.client(service, region_name=os.getenv('AWS_REGION', 'us-west-2'))