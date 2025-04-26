import os
import threading
import asyncio
import time
import logging
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO
from tools.retry_utils import retry_async, retry_sync
from live_transcriber.live_transcriber import LiveTranscriber
from rag_chat.rag import RAGPipeline, WebSearcher, ConversationalModel
from rag_chat.chat import Chatbot
from tts.tts import PollyTTS
from agent.action_decompose import ActionDecomposer
from task_classification.task_classification import TaskClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 🔥 可取消的處理任務狀態
current_task = None
current_task_lock = threading.Lock()

HTML = '''
<!doctype html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <title>Robot Emotions 🤖</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      margin: 0;
      padding: 0;
      background-color: #0b0c10;
      color: #c5c6c7;
      font-family: 'Noto Sans TC', sans-serif;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    #left, #right {
      width: 50%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    #left {
      background-color: #0b0c10;
      padding: 20px;
    }
    #right {
      background-color: #1f2833;
      overflow-y: auto;
      padding: 30px 20px;
    }
    #expression {
      width: 85%;
      max-width: 600px;
      border-radius: 20px;
      background-color: #1f2833;
      padding: 10px;
    }
    #status {
      margin-top: 20px;
      font-size: 20px;
      color: #66fcf1;
      font-weight: bold;
      text-align: center;
    }
    #chat_log {
      width: 100%;
      max-width: 700px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .chat_entry {
      background: #45a29e;
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
    }
    .user_query {
      font-size: 20px;
      font-weight: 700;
      color: #0b0c10;
    }
    .bot_response {
      font-size: 18px;
      color: #0b0c10;
      margin-top: 12px;
    }
    #player {
      display: none;
    }
    #click_to_start {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.95);
      color: #66fcf1;
      font-size: 28px;
      font-weight: bold;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      cursor: pointer;
    }
  </style>
</head>

<body>

<div id="click_to_start">🔈 點一下開始</div>

<div id="left">
  <img id="expression" src="/static/animations/wakeup.svg" />
  <div id="status">🎤 等待開始錄音...</div>
  <audio id="player" controls></audio>
</div>

<div id="right">
  <div id="chat_log"></div>
</div>

<script>
  const socket = io();
  const expr = document.getElementById('expression');
  const status = document.getElementById('status');
  const player = document.getElementById('player');
  const chatLog = document.getElementById('chat_log');
  const clickLayer = document.getElementById('click_to_start');

  let latestUserQuery = null;

  window.onload = () => {
    clickLayer.addEventListener('click', () => {
      socket.emit('start_listening');
      status.innerText = '🎤 錄音中...';
      player.play().catch(e => console.log("🔕 初次播放失敗（正常）"));
      clickLayer.style.display = 'none';
    });
  };

  socket.on('expression', (path) => {
    expr.src = path;
  });

  socket.on('audio_url', (url) => {
    console.log("🔔 收到新的音檔 URL，開始播放");
    expr.src = '/static/animations/speaking.gif';
    player.pause();
    player.src = url;
    player.load();
    player.play()
      .then(() => console.log("🔊 音訊播放成功！"))
      .catch((err) => {
        console.error("❌ 播放失敗：", err);
        status.innerText = '⚠️ 無法播放音訊，請檢查瀏覽器設定';
      });

    player.onended = () => {
      console.log("🔕 音訊播放完畢，自動切回 idle");
      expr.src = '/static/animations/idle.gif';
    };
  });

  socket.on('status', (msg) => {
    status.innerText = msg;
  });

  socket.on('user_query', (text) => {
    latestUserQuery = text;
  });

  socket.on('text_response', (text) => {
    const entry = document.createElement('div');
    entry.className = 'chat_entry';
    entry.innerHTML = `
      <div class="user_query">🧑 ${latestUserQuery || ''}</div>
      <div class="bot_response">🤖 ${text}</div>
    `;
    chatLog.appendChild(entry);
    chatLog.scrollTop = chatLog.scrollHeight;
  });
</script>

</body>
</html>


'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/history_result/<filename>')
def get_audio(filename):
    return send_from_directory('history_result', filename)

# async def handle_text(text: str):
#     try:
#         logger.info(f"[handle_text] 收到完整文字：{text}")
#         socketio.emit('status', f"📝 偵測到文字：{text}")
#         socketio.emit('user_query', text)

#         tc = TaskClassifier()
#         task_type, _ = tc.classify_task(text)
#         logger.info(f"[handle_text] 任務分類結果：{task_type}")

#         socketio.emit('expression', '/static/animations/thinking.gif')

#         audio_path = None
#         generated_text = None

#         ts = time.strftime('%Y%m%d_%H%M%S')

#         if task_type == "聊天":
#             chat_model = Chatbot(model_id="anthropic.claude-3-haiku-20240307-v1:0")
#             generated_text = chat_model.chat(text)
#             audio_path = f"./history_result/output_chat_{ts}.mp3"
#             PollyTTS().synthesize(generated_text, audio_path)

#         elif task_type == "查詢":
#             ws = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
#             model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
#             pipeline = RAGPipeline(web_searcher=ws, model=model)
#             generated_text = pipeline.answer(text)
#             audio_path = f"./history_result/output_search_{ts}.mp3"
#             PollyTTS().synthesize(generated_text, audio_path)

#         elif task_type == "行動":
#             ad = ActionDecomposer()
#             generated_text = ad.decompose(text)
#             audio_path = None

#         if generated_text:
#             socketio.emit('text_response', generated_text)

#         if audio_path and Path(audio_path).exists():
#             logger.info(f"[handle_text] 音檔生成完成：{audio_path}")
#             audio_url = f"/history_result/{os.path.basename(audio_path)}"
#             socketio.emit('expression', '/static/animations/speaking.gif')
#             socketio.emit('audio_url', audio_url)

#         socketio.emit('status', '✅ 已完成。')

#     except Exception as e:
#         logger.error(f"[handle_text] 發生錯誤：{e}")

async def handle_text(text: str):
    try:
        logger.info(f"[handle_text] 收到完整文字：{text}")
        socketio.emit('status', f"📝 偵測到文字：{text}")
        socketio.emit('user_query', text)

        tc = TaskClassifier()
        task_type, _ = retry_sync(retries=3, delay=1)(tc.classify_task)(text)
        logger.info(f"[handle_text] 任務分類結果：{task_type}")

        socketio.emit('expression', '/static/animations/thinking.gif')

        audio_path = None
        generated_text = None

        ts = time.strftime('%Y%m%d_%H%M%S')

        if task_type == "聊天":
            chat_model = Chatbot(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            generated_text = retry_sync(retries=3, delay=1)(chat_model.chat)(text)
            audio_path = f"./history_result/output_chat_{ts}.mp3"
            retry_sync(retries=3, delay=1)(PollyTTS().synthesize)(generated_text, audio_path)

        elif task_type == "查詢":
            ws = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
            model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            pipeline = RAGPipeline(web_searcher=ws, model=model)
            generated_text = retry_sync(retries=3, delay=1)(pipeline.answer)(text)  # ✅ 改這裡
            audio_path = f"./history_result/output_search_{ts}.mp3"
            retry_sync(retries=3, delay=1)(PollyTTS().synthesize)(generated_text, audio_path)

        elif task_type == "行動":
            ad = ActionDecomposer()
            generated_text = retry_sync(retries=3, delay=1)(ad.decompose)(text)
            audio_path = None

        if generated_text:
            socketio.emit('text_response', generated_text)

        if audio_path and Path(audio_path).exists():
            logger.info(f"[handle_text] 音檔生成完成：{audio_path}")
            audio_url = f"/history_result/{os.path.basename(audio_path)}"
            socketio.emit('expression', '/static/animations/speaking.gif')
            socketio.emit('audio_url', audio_url)

        socketio.emit('status', '✅ 已完成。')

    except Exception as e:
        logger.error(f"[handle_text] 發生錯誤：{e}")



async def cancellable_socket_handle_text(text: str):
    global current_task

    with current_task_lock:
        # 先取消舊的
        if current_task and not current_task.done():
            logger.info("[cancellable_socket_handle_text] 取消上一個任務...")
            current_task.cancel()

        # 再開新的
        loop = asyncio.get_running_loop()
        current_task = loop.create_task(handle_text(text))

def run_transcriber():
    logger.info("[run_transcriber] 啟動 LiveTranscriber！")
    with app.app_context():
        attempt = 0
        max_attempts = 2
        while attempt < max_attempts:
            try:
                transcriber = LiveTranscriber(region="us-west-2", callback=cancellable_socket_handle_text)
                asyncio.run(transcriber.start())
                break
            except Exception as e:
                attempt += 1
                logger.error(f"[run_transcriber] LiveTranscriber 連線失敗（第 {attempt} 次），錯誤: {e}")
                if attempt >= max_attempts:
                    logger.error("[run_transcriber] 已達最大重試次數，放棄連線。")
                else:
                    logger.info("[run_transcriber] 等待 2 秒後重試...")
                    time.sleep(1)

@socketio.on('start_listening')
def handle_start():
    threading.Thread(target=run_transcriber, daemon=True).start()

if __name__ == '__main__':
    os.makedirs('history_result', exist_ok=True)
    socketio.run(app, host='0.0.0.0', port=5000)