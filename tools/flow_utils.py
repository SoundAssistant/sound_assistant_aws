import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_chat.rag import RAGPipeline , WebSearcher , ConversationalModel
from rag_chat.chat import Chatbot
from tts.tts import PollyTTS
from cache_utils import get_cache

def search_flow(query):
    web_searcher = WebSearcher(max_results=3, search_depth="advanced")
    # retriever = Retriever("YOUR_KB_ID", number_of_results=3)  
    model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    pipeline = RAGPipeline(
        # retriever=retriever,  
        web_searcher=web_searcher,
        model=model
    )
    answer = pipeline.answer(query)
    tts_model = PollyTTS()
    tts_model.synthesize(answer, "./history_result/output_search.mp3")
    print(answer)

def chat_flow(query):
    chat_model = Chatbot(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    response = chat_model.chat(query)
    tts_model = PollyTTS()
    tts_model.synthesize(response, "./history_result/output_chat.mp3")
    print(response)

if __name__ == "__main__":
    cache = get_cache()
    cache.clear()
    chat_flow("目前台積電的最新股價你記錄到哪天的")
    cache.clear()
    search_flow("目前台積電的最新股價你記錄到哪天的")
    
    