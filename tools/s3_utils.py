import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.client_utils import get_s3_client


s3 = get_s3_client('s3')


def upload_file_to_s3(local_path: str, bucket: str, key: str) -> bool:
    """
    將本地檔案上傳到 S3。
    local_path: 檔案的本地路徑
    bucket: S3 的 Bucket 名稱
    key: S3 中的完整檔案路徑（含資料夾）
    """
    try:
        s3.upload_file(local_path, bucket, key)
        print(f"✅ 成功上傳至 S3: s3://{bucket}/{key}")
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"❌ 上傳失敗: {e}")
        return False


def list_objects_in_bucket(bucket: str, prefix: str = "") -> list:
    """
    列出指定 S3 bucket + prefix 下的所有檔案。
    回傳格式為檔案 key 的 list。
    """
    try:
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

        keys = []
        for page in page_iterator:
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    except (BotoCoreError, ClientError) as e:
        print(f"❌ 讀取 S3 內容失敗: {e}")
        return []
