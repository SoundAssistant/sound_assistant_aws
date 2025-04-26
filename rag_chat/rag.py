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
from tools.cache_utils import get_cache

class WebSearcher:
    def __init__(self,
                 api_key: str = None,
                 max_results: int = 3,
                 search_depth: str = "basic",
                 use_top_only: bool = False): 
        load_dotenv()
        api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Please set the TAVILY_API_KEY environment variable")
        self.client = TavilyClient(api_key)
        self.max_results = max_results
        self.search_depth = search_depth
        self.use_top_only = use_top_only 

    def get_context(self, query: str) -> str:
        raw_result = self.client.get_search_context(
            query=query,
            max_results=self.max_results,
            search_depth=self.search_depth
        )
        try:
            result = json.loads(raw_result)

            if self.use_top_only:
                result = result[:1]  

            formatted_contexts = [
                f"資料來自：{item['url']}\n{item['content']}" for item in result
                if 'content' in item and 'url' in item
            ]
            return "\n\n".join(formatted_contexts)
        except Exception as e:
            print("Error parsing Tavily search result:", e)
            return "無法解析搜尋結果。"


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
        以下是您需要參考的資料內容：
        <context>
        {chr(10).join(contexts)}
        </context>

        根據上方資料，請以「摘要」的形式回答以下問題：
        <question>
        {query}
        </question>

        回答要求：
        - 請以 50 到 70 字之間總結回答
        - 不要逐字抄寫內容，要用自己的話整理重點
        - 如果資料不足，請回答：「根據目前的資料無法回答此問題。」
        - 不要附上資料來源或連結
        - 使用簡明扼要、直接了當的中文回答
        """


class ConversationalModel:
    def __init__(self, model_id: str, temperature: float = 0.1, top_k: int = 200):
        self.client = get_bedrock_runtime_client()
        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.system_prompts = [
            {
                "text": (
                    "You are a strict question answering assistant. "
                    "You must answer ONLY based on the provided <context>. "
                    "Summarize the answer concisely in 50 to 70 words. "
                    "If the context does not contain the answer, reply with: '根據目前的資料無法回答此問題。' "
                    "Do not copy long sentences from the context directly. "
                    "Please DO NOT include any source links or citations in your answer."
                )
            }
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
        self.cache = get_cache()

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
                self.cache.add_to_cache(query, resp['content'][0]['text'])
                return resp['content'][0]['text']
            except ClientError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay * (2 ** attempt))

if __name__ == "__main__":
    web_searcher = WebSearcher(max_results=3, search_depth="advanced",use_top_only=True )
    # retriever = Retriever("YOUR_KB_ID", number_of_results=3)  
    model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    pipeline = RAGPipeline(
        # retriever=retriever,  
        web_searcher=web_searcher,
        model=model
    )
    answer = pipeline.answer("請給我最新的AWS重大新聞 並告訴我這是哪天的")
    print(answer)

