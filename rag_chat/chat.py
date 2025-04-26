import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.cache_utils import get_cache
from botocore.exceptions import ClientError
import boto3
import pandas as pd
import json

class Chatbot:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.bedrock = boto3.client("bedrock-runtime")
        self.cache = get_cache()

    def generate_response(self, query: str) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"請以不超過 80 字的方式回答以下問題：{query}"}
            ]
        }
    ]

        }

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def chat(self, query: str) -> str:
        try:
            # 從 cache 拿，如果沒有就用 generate_response 並加進 cache
            response = self.cache.get_or_generate_response(
                query, self.generate_response
            )
            return response
        except ClientError as e:
            print(f"API ERROR: {e}")
            return "目前伺服器有問題，請稍後再試。"

        
if __name__ == "__main__":

    chat_model = Chatbot("anthropic.claude-3-haiku-20240307-v1:0")
    answer = chat_model.chat("請給我最新的AWS重大新聞")
    print(answer)