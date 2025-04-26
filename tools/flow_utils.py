import os
import threading
import asyncio
import time
import logging
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO

# 引入你的後端邏輯
from live_transcriber.live_transcriber import LiveTranscriber
from rag_chat.rag import RAGPipeline, WebSearcher, ConversationalModel
from rag_chat.chat import Chatbot
from tts.tts import PollyTTS
from agent.action_decompose import ActionDecomposer
from task_classification.task_classification import TaskClassifier

# 初始化 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

HTML = '''
<!doctype html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <title>Robot Emotions 🧙‍♂️</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
  <style>
    body {
      margin: 0;
      padding: 0;
      display: flex;
      height: 100vh;
      overflow: hidden;
      background-color: #000000; /* 全黑背景 */
      color: white;
      font-family: Arial, sans-serif;
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
      background-color: #000000;
    }
    #right {
      background-color: #111111;
      overflow-y: auto;
      padding: 20px;
      text-align: left;
    }
    #expression {
      width: 80%;
      max-width: 600px;
    }
    #status {
      margin-top: 20px;
      font-size: 24px;
      color: #00ffcc;
    }
    #chat_log {
      width: 100%;
    }
    .chat_entry {
      margin-bottom: 30px;
    }
    .user_query {
      font-size: 28px;
      font-weight: bold;
      color: #ffcc00;
    }
    .bot_response {
      font-size: 28px;
      color: #00bfff;
      margin-top: 10px;
      margin-left: 20px;
    }
    #player {
      display: none; /* 隱藏 audio 控制器 */
    }
  </style>
</head>
<body>

<div id="left">
  <img id="expression" src="/static/animations/wakeup.svg" />
  <div id="status">🎤 等待開始錄音...</div>
  <audio id="player" controls></audio>
</div>

<div id="right">
  <div id="chat_log">
    <!-- 這裡會自動插入聊天訊息 -->
  </div>
</div>

<script>
  const socket = io();
  const expr = document.getElementById('expression');
  const status = document.getElementById('status');
  const player = document.getElementById('player');
  const chatLog = document.getElementById('chat_log');

  let latestUserQuery = null; // 暫存使用者問的文字

  window.onload = () => {
    socket.emit('start_listening');
    status.innerText = '🎤 錄音中...';
  };

  socket.on('expression', (path) => {
    expr.src = path;
  });

  socket.on('audio_url', (url) => {
    player.src = url;
    player.play();
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
    chatLog.scrollTop = chatLog.scrollHeight; // 自動捲到最底
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

async def socket_handle_text(text: str):
    try:
        logger.info(f"[socket_handle_text] 收到完整文字：{text}")
        socketio.emit('status', f"📝 偵測到文字：{text}")
        socketio.emit('user_query', text)   # ✅ 記錄使用者說的內容

        # 先任務分類
        tc = TaskClassifier()
        task_type, _ = tc.classify_task(text)
        logger.info(f"[socket_handle_text] 任務分類結果：{task_type}")

        socketio.emit('expression', '/static/animations/thinking.gif')

        wav = None
        generated_text = None

        if task_type == "聊天":
            logger.info("[socket_handle_text] 走 chat flow")
            chat_model = Chatbot(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            generated_text = chat_model.chat(text)
            logger.info(f"[socket_handle_text] 回應內容：{generated_text}")

            tts = PollyTTS()
            ts = time.strftime('%Y%m%d_%H%M%S')
            wav = f"./history_result/output_chat_{ts}.wav"
            tts.synthesize(generated_text, wav)

        elif task_type == "查詢":
            logger.info("[socket_handle_text] 走 search flow")
            ws = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
            model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            pipeline = RAGPipeline(web_searcher=ws, model=model)
            generated_text = pipeline.answer(text)
            logger.info(f"[socket_handle_text] 查詢結果：{generated_text}")

            tts = PollyTTS()
            ts = time.strftime('%Y%m%d_%H%M%S')
            wav = f"./history_result/output_search_{ts}.wav"
            tts.synthesize(generated_text, wav)

        elif task_type == "行動":
            logger.info("[socket_handle_text] 走 action flow")
            ad = ActionDecomposer()
            generated_text = ad.decompose(text)
            wav = None

        if generated_text:
            socketio.emit('text_response', generated_text)

        if wav and Path(wav).exists():
            logger.info(f"[socket_handle_text] 音檔生成完成：{wav}")
            socketio.emit('expression', '/static/animations/speaking.gif')
            audio_url = f"/history_result/{os.path.basename(wav)}"
            logger.info(f"[socket_handle_text] 音檔URL: {audio_url}")
            socketio.emit('audio_url', audio_url)

        socketio.emit('expression', '/static/animations/idle.gif')
        socketio.emit('status', '✅ 已完成。')

    except Exception as e:
        logger.error(f"[socket_handle_text] 發生錯誤：{e}")

def run_transcriber():
    logger.info("[run_transcriber] 啟動 LiveTranscriber！")
    with app.app_context():
        transcriber = LiveTranscriber(region="us-west-2", callback=socket_handle_text)
        try:
            asyncio.run(transcriber.start())
        except Exception as e:
            logger.error(f"[run_transcriber] 發生錯誤：{e}")

@socketio.on('start_listening')
def handle_start():
    threading.Thread(target=run_transcriber, daemon=True).start()

if __name__ == '__main__':
    os.makedirs('history_result', exist_ok=True)
    socketio.run(app, host='0.0.0.0', port=5000)
