import os
import json
import time
from typing import List, Dict
from botocore.exceptions import ClientError
import sys
import boto3
from tavily import TavilyClient  
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.client_utils import get_bedrock_runtime_client

class WebSearcher:
    def __init__(self,
                 api_key: str = None,
                 max_results: int = 5,
                 search_depth: str = "basic"):
        
        load_dotenv()
        api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Please set the TAVILY_API_KEY environment variable")
        self.client = TavilyClient(api_key)  
        self.max_results = max_results
        self.search_depth = search_depth

    def get_context(self, query: str) -> str:
        return self.client.get_search_context(
            query=query,
            max_results=self.max_results,
            search_depth=self.search_depth
        )

class Retriever:
    def __init__(self, knowledge_base_id: str, number_of_results: int = 5):
        self.agent_client = get_bedrock_runtime_client()
        self.knowledge_base_id = knowledge_base_id
        self.number_of_results = number_of_results

    def retrieve(self, query: str) -> List[str]:
        resp = self.agent_client.retrieve(
            retrievalQuery={'text': query},
            knowledgeBaseId=self.knowledge_base_id,
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': self.number_of_results
                }
            }
        )
        return [r['content']['text'] for r in resp['retrievalResults']]

class PromptBuilder:
    @staticmethod
    def build_prompt(contexts: List[str], query: str) -> str:
        return f"""
            Here is the context to reference:
            <context>
            {chr(10).join(contexts)}
            </context>

            Referencing the context, answer the user question
            <question>
            {query}
            </question>
            """

class ConversationalModel:
    def __init__(self, model_id: str, temperature: float = 0.5, top_k: int = 200):
        self.client = get_bedrock_runtime_client()
        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.system_prompts = [
            {"text": "You are a Question and Answering assistant. Answer based on provided context."}
        ]

    def converse(self, messages: List[Dict]) -> Dict:
        response = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=self.system_prompts,
            inferenceConfig={"temperature": self.temperature},
            additionalModelRequestFields={"top_k": self.top_k}
        )
        return response['output']['message']

class RAGPipeline:
    def __init__(self,
                 # retriever: Retriever,   # 目前因為沒有kb所以先不用
                 web_searcher: WebSearcher,
                 model: ConversationalModel):
        # self.retriever = retriever  # 目前因為沒有kb所以先不用
        self.web_searcher = web_searcher
        self.model = model
        self.messages: List[Dict] = []

    def answer(self, query: str) -> str:
        web_ctx = self.web_searcher.get_context(query)
        # vector_ctxs = self.retriever.retrieve(query)  # 目前因為沒有kb所以先不用
        # all_ctx = [web_ctx] + vector_ctxs  # 目前因為沒有kb所以先不用
        all_ctx = [web_ctx]  # 僅使用 web context

        prompt = PromptBuilder.build_prompt(all_ctx, query)
        user_msg = {"role": "user", "content": [{"text": prompt}]}
        self.messages.append(user_msg)

        max_retries, delay = 5, 1
        for attempt in range(max_retries):
            try:
                resp = self.model.converse(self.messages)
                self.messages.append(resp)
                return resp['content'][0]['text']
            except ClientError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay * (2 ** attempt))

if __name__ == "__main__":
    web_searcher = WebSearcher(max_results=3, search_depth="advanced")
    # retriever = Retriever("YOUR_KB_ID", number_of_results=3)  # 暫時不使用
    model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    pipeline = RAGPipeline(
        # retriever=retriever,  # 暫時不使用
        web_searcher=web_searcher,
        model=model
    )
    answer = pipeline.answer("請給我最新的AWS重大新聞")
    print(answer)

