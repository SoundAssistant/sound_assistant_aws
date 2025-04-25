import time
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
import boto3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.s3_utils import upload_file_to_s3
from tools.client_utils import get_bedrock_client,get_bedrock_runtime_client


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class LFUSlot:
    def __init__(self, query: str, response: str, embedding: np.ndarray, ttl: int):
        self.query = query
        self.response = response
        self.embedding = embedding
        self.create_date = time.time()
        self.ttl = ttl
        self.usage_count = 0


class InMemorySemanticCache:
    def __init__(self, similarity_threshold: float = 0.7, max_cache_size: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache: List[LFUSlot] = []

        self.session_log: List[Tuple[str, str]] = []

        self.model_id = "amazon.titan-embed-text-v2:0"
        self.bedrock = get_bedrock_runtime_client("bedrock-runtime")

    def get_embedding(self, text: str) -> np.ndarray:
        body = {"inputText": text}
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return np.array(result["embedding"])

    def add_to_cache(self, query: str, response: str, ttl: int = 3600):
        embedding = self.get_embedding(query)
        if len(self.cache) >= self.max_cache_size:
            self.cache.sort(key=lambda x: x.usage_count)
            self.cache.pop(0)
        self.cache.append(LFUSlot(query, response, embedding, ttl))

    def query_cache(self, query: str, k: int = 3) -> Optional[str]:
        if not self.cache:
            return None

        q_emb = self.get_embedding(query)
        scored_entries = [
            (cosine_similarity(q_emb, item.embedding), i)
            for i, item in enumerate(self.cache)
        ]
        scored_entries.sort(reverse=True)
        for score, idx in scored_entries[:k]:
            if score >= self.similarity_threshold:
                self.cache[idx].usage_count += 1
                return self.cache[idx].response
        return None

    def get_or_generate_response(self, query: str, fallback_generator) -> str:
        cached = self.query_cache(query)
        if cached:
            self.session_log.append((query, cached))
            return cached

        new_response = fallback_generator(query)
        self.add_to_cache(query, new_response)
        self.session_log.append((query, new_response))
        return new_response

    def save_session_to_txt_and_upload(self, filename_prefix: str, s3_bucket: str, s3_key_prefix: str = ""):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        local_filename = f"{filename_prefix}_{timestamp}.txt"

        with open(local_filename, "w", encoding="utf-8") as f:
            for q, r in self.session_log:
                f.write(f"User: {q}\nBot: {r}\n{'-' * 40}\n")

        s3_key = f"{s3_key_prefix}{local_filename}"
        success = upload_file_to_s3(local_filename, s3_bucket, s3_key)

        if success:
            self.session_log.clear()


def dummy_generator(query: str) -> str:
    return f"這是 LLM 回答「{query}」"

if __name__ == '__main__':
    cache = InMemorySemanticCache()

    cache.get_or_generate_response("你是誰？", dummy_generator)
    cache.get_or_generate_response("今天幾號？", dummy_generator)

    cache.save_session_to_txt_and_upload(
        filename_prefix="./history_result/chat_log",
        s3_bucket="soundassistant",
        s3_key_prefix="chatlogs/"
    )
