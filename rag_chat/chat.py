from cache_tools import cache
import rag as rag
from botocore.exceptions import ClientError
import boto3
import pandas as pd

class Chatbot:
    def __init__(self, model_id: str, s3_bucket: str, s3_key_prefix: str = ""):
        self.model_id = model_id
        self.s3_bucket = s3_bucket
        self.s3_key_prefix = s3_key_prefix
        self.bedrock = boto3.client("bedrock-runtime")
        self.cache = cache.InMemorySemanticCache()
        self.rag = rag.RAG(self.bedrock, self.model_id)

    def chat(self, query: str) -> str:
        try:
            response = self.cache.get_or_generate_response(
                query, lambda q: self.rag.generate_response(q)
            )
            return response
        except ClientError as e:
            print(f"❌ Bedrock API 請求失敗: {e}")
            return "抱歉，我目前無法回答您的問題。請稍後再試。"

    def save_session_to_s3(self, filename_prefix: str):
        self.cache.save_session_to_txt_and_upload(
            filename_prefix, self.s3_bucket, self.s3_key_prefix
        )
        print(f"✅ 保存会话到 S3: s3://{self.s3_bucket}/{self.s3_key_prefix}")