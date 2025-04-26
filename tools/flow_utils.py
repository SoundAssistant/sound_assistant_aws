import asyncio
import sys
<<<<<<< HEAD
import json
=======
import os
import time
>>>>>>> dcb2fccc9713537d3170d1e7ba43be6b1de57c7c
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_chat.rag import RAGPipeline, WebSearcher, ConversationalModel
from rag_chat.chat import Chatbot
from tts.tts import PollyTTS
from cache_utils import get_cache
<<<<<<< HEAD
from agent.action_decompose import ActionDecomposer
=======
from task_classification.task_classification import TaskClassifier
from live_transcriber.live_transcriber import LiveTranscriber
>>>>>>> dcb2fccc9713537d3170d1e7ba43be6b1de57c7c

def search_flow(query: str):
    web_searcher = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
    model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    pipeline = RAGPipeline(web_searcher=web_searcher, model=model)
    answer = pipeline.answer(query)
    
    tts_model = PollyTTS()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    tts_model.synthesize(answer, f"./history_result/output_search_{timestamp}.mp3")
    print(f"🔎 搜尋結果：{answer}")

def chat_flow(query: str):
    chat_model = Chatbot(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    response = chat_model.chat(query)

    tts_model = PollyTTS()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    tts_model.synthesize(response, f"./history_result/output_chat_{timestamp}.mp3")
    print(f"💬 聊天回應：{response}")

def task_flow(query: str) -> str:
    task_classifier = TaskClassifier()
    task_type, task_description = task_classifier.classify_task(query)
    return task_type

async def handle_text(text: str):
    print(f"🎤 偵測到文字：{text}")

    task_type = task_flow(text)

    if task_type == "聊天":
        chat_flow(text)
    elif task_type == "查詢":
        search_flow(text)
    elif task_type == "行動":
        print("🛠️ 行動任務，目前尚未實作 flow")
    else:
        print(f"❓ 未知任務類型：{task_type}")

async def stt_flow():
    transcriber = LiveTranscriber(region="us-west-2", callback=handle_text)
    await transcriber.start()

def main_flow():
    try:
        asyncio.run(stt_flow())
    except KeyboardInterrupt:
        print("\n🎤 偵測結束（使用者中斷）")
    except Exception as e:
        print(f"⚠️ 發生錯誤：{e}")

def action_flow(query):
    decomposer = ActionDecomposer(model_id = "anthropic.claude-3-sonnet-20240229-v1:0")
    response = decomposer.decompose(query)
    print(response)

if __name__ == "__main__":
    cache = get_cache()
    cache.clear()
<<<<<<< HEAD
    chat_flow("目前台積電的最新股價為多少")
    cache.clear()
    search_flow("目前台積電的最新股價為多少")
    cache.clear()
    action_flow("幫我把這杯水倒入杯子中，然後跟使用者說「完成了」。")
    
    
=======

    print("🚀 啟動語音助理系統！請開始說話...")
    main_flow()
>>>>>>> dcb2fccc9713537d3170d1e7ba43be6b1de57c7c
