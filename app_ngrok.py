import os
import threading
import asyncio
import time
import logging
import base64
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory, url_for
from urllib.parse import urljoin
from flask import request
from flask_socketio import SocketIO
from tools.retry_utils import retry_sync
from rag_chat.rag import RAGPipeline, WebSearcher, ConversationalModel
from rag_chat.chat import Chatbot
from tts.tts import PollyTTS
from agent.action_decompose import ActionDecomposer
from task_classification.task_classification import TaskClassifier
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# --- 環境初始化 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.config['SERVER_NAME'] = '0747-34-222-37-198.ngrok-free.app'
socketio = SocketIO(app, cors_allowed_origins="*")

current_task = None
current_task_lock = threading.Lock()

# --- 啟動時檢查 ffmpeg ---
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
except Exception:
    logger.error("❌ 找不到 ffmpeg，請安裝 ffmpeg。")
    raise

# --- Transcript Handler ---
class MyTranscriptHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                text = result.alternatives[0].transcript.strip()
                if text:
                    logger.info(f"[process_audio_file] 轉出文字：{text}")
                    await cancellable_socket_handle_text(text)

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
      position: relative;
    }
    #left {
      background-color: #0b0c10;
      padding: 20px;
    }
    #right {
      background-color: #1f2833;
      overflow-y: auto;
      padding: 30px 20px 30px 20px;
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
      padding-top: 30px;
      padding-bottom: 50px;
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
    #volume_bar {
      width: 80%;
      height: 10px;
      background: #ccc;
      margin-top: 15px;
      border-radius: 5px;
      overflow: hidden;
    }
    #volume_fill {
      height: 100%;
      width: 0%;
      background: #66fcf1;
      transition: width 0.1s;
    }
  </style>
</head>

<body>

<div id="click_to_start">🔈 點一下開始</div>

<div id="left">
  <img id="expression" src="/static/animations/wakeup.svg" />
  <div id="status">🎤 等待開始錄音...</div>
  <div id="volume_bar"><div id="volume_fill"></div></div>
  <audio id="player" controls></audio>
</div>

<div id="right">
  <div id="chat_log"></div>
</div>

<script>
const socket = io();
const expr = document.getElementById('expression');
const status = document.getElementById('status');
const volumeFill = document.getElementById('volume_fill');
const player = document.getElementById('player');
const chatLog = document.getElementById('chat_log');
const clickLayer = document.getElementById('click_to_start');

let latestUserQuery = null;
let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let stream;
let isRecording = false;
let recordingStartTime = null;
let silenceStart = null;
let weakNoiseStart = null;
let backgroundVolumes = [];
let hasRecordedOnce = false; 

const baseThreshold = 0.08;             // 基本啟動門檻
let dynamicThreshold = baseThreshold;    // 動態啟動門檻
const silenceThreshold = 0.02;           // 判定無聲
const silenceDelay = 2000;               // 錄音中無聲多久停止錄音（毫秒）
const maxRecordingTime = 12000;           // 錄音最大時長（毫秒）
const weakNoiseIgnoreTime = 3000;         // 小聲雜訊超過多久忽略（毫秒）

window.onload = () => {
  clickLayer.addEventListener('click', async () => {
    try {
      await prepareMicrophone();
      clickLayer.style.display = 'none';
    } catch (err) {
      console.error('⚠️ 無法啟動錄音：', err);
      status.innerText = '❌ 無法啟動錄音';
    }
  });
};

async function prepareMicrophone() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);

  mediaRecorder.addEventListener('dataavailable', event => {
    audioChunks.push(event.data);
  });

  mediaRecorder.addEventListener('stop', async () => {
    hasRecordedOnce = true;
    if (audioChunks.length > 0) {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];

      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];
        socket.emit('audio_blob', base64Audio);
        status.innerText = '📨 上傳音訊中...';
      };
      reader.readAsDataURL(audioBlob);
    }
    setTimeout(startListening, 500);
  });

  startListening();
}

function startListening() {
  isRecording = false;
  recordingStartTime = null;
  silenceStart = null;
  weakNoiseStart = null;
  backgroundVolumes = [];
  audioChunks = [];
  status.innerText = '👂 正在靜音監聽中...';
  
  // ⭐ 重點：第一次用 wakeup.svg，以後用 thinking.gif
  if (!hasRecordedOnce) {
    expr.src = '/static/animations/wakeup.svg';
  } else {
    expr.src = '/static/animations/thinking.gif';
  }

  monitorVolume();
}

function monitorVolume() {
  const dataArray = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(dataArray);

  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const normalized = (dataArray[i] - 128) / 128;
    sum += normalized * normalized;
  }
  const volume = Math.sqrt(sum / dataArray.length);

  // 更新音量條
  const volumePercentage = Math.min(100, Math.floor(volume * 300));
  volumeFill.style.width = volumePercentage + '%';

  const now = Date.now();

  // --- 背景音量統計 (只在待機時做) ---
  if (!isRecording) {
    backgroundVolumes.push(volume);
    if (backgroundVolumes.length > 100) backgroundVolumes.shift();

    const avgBackground = backgroundVolumes.reduce((a, b) => a + b, 0) / backgroundVolumes.length;
    if (avgBackground > 0.05) {
      dynamicThreshold = Math.min(0.15, baseThreshold + (avgBackground - 0.05));
    } else {
      dynamicThreshold = baseThreshold;
    }
  }

  // --- 小聲雜訊忽略 ---
  if (!isRecording) {
    if (volume > silenceThreshold && volume < dynamicThreshold) {
      if (!weakNoiseStart) weakNoiseStart = now;
      if (now - weakNoiseStart > weakNoiseIgnoreTime) {
        console.log('💤 小聲雜訊超過3秒，忽略');
        weakNoiseStart = null;
        backgroundVolumes = [];
      }
    } else {
      weakNoiseStart = null;
    }
  }

  // --- 錄音邏輯 ---
  if (!isRecording) {
    if (volume > dynamicThreshold) {
      console.log('🎙️ 偵測到說話，開始錄音！');
      mediaRecorder.start();
      recordingStartTime = now;
      silenceStart = null;
      isRecording = true;
      status.innerText = '🎤 錄音中...';
      expr.src = '/static/animations/thinking.gif';
    }
  } else {
    if (volume > silenceThreshold) {
      silenceStart = null;
    } else {
      if (!silenceStart) silenceStart = now;
      if (now - silenceStart > silenceDelay) {
        console.log('🛑 錄音中偵測到靜音超過2秒，停止錄音');
        mediaRecorder.stop();
        return;
      }
    }
    if (now - recordingStartTime > maxRecordingTime) {
      console.log('⏰ 錄音超過12秒，強制停止');
      mediaRecorder.stop();
      return;
    }
  }

  requestAnimationFrame(monitorVolume);
}

// --- 處理 server 回傳訊息 ---
socket.on('expression', (path) => {
  expr.src = path;
});

socket.on('audio_url', (url) => {
  expr.src = '/static/animations/speaking.gif';
  player.pause();
  player.src = url;   // 👈 直接用，不要自己亂補 window.location.href 或 urljoin 了
  player.load();
  player.play().catch(err => console.error("❌ 播放失敗", err));
  player.onended = () => {
    expr.src = '/static/animations/idle.gif';
    if (player.src.includes("/history_result/")) {
      const filename = player.src.split("/history_result/")[1];
      socket.emit('delete_audio', filename);
    }
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
@socketio.on('delete_audio')
def delete_audio(filename):
    try:
        path = os.path.join('history_result', filename)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"[delete_audio] 已刪除檔案：{path}")
    except Exception as e:
        logger.error(f"[delete_audio] 刪除檔案失敗：{e}")
        
# --- 音訊處理 ---
@socketio.on('audio_blob')
def handle_audio_blob(base64_audio):
    logger.info("[handle_audio_blob] 收到音訊 blob，準備轉文字...")
    
    # ⭐ 新增：收到音訊後馬上切換成 thinking.gif
    socketio.emit('expression', '/static/animations/thinking.gif')

    try:
        audio_data = base64.b64decode(base64_audio)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_audio_file(tmp_file_path))
            loop.close()

        threading.Thread(target=run_in_thread).start()

    except Exception as e:
        logger.error(f"[handle_audio_blob] 音訊處理失敗：{e}")


async def process_audio_file(file_path):
    try:
        pcm_path = file_path.replace('.webm', '.wav')
        subprocess.run(f"ffmpeg -y -i {file_path} -ac 1 -ar 16000 -f wav {pcm_path}", shell=True, check=True)

        with open(pcm_path, 'rb') as f:
            pcm_data = f.read()

        client = TranscribeStreamingClient(region="us-west-2")
        stream = await client.start_stream_transcription(
            language_code="zh-TW",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        chunk_size = 6400  
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i+chunk_size]
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await asyncio.sleep(0.1)

        await stream.input_stream.end_stream()

        handler = MyTranscriptHandler(stream.output_stream)
        async for event in stream.output_stream:
            await handler.handle_transcript_event(event)

    except Exception as e:
        logger.error(f"[process_audio_file] 音訊處理失敗：{e}")

# --- 任務處理 ---
async def handle_text(text: str):
    try:
        logger.info(f"[handle_text] 收到完整文字：{text}")
        socketio.emit('status', f"📝 偵測到文字：{text}")
        socketio.emit('user_query', text)

        task_classifier = TaskClassifier()
        task_type, _ = retry_sync(retries=3, delay=1)(task_classifier.classify_task)(text)
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
            web_searcher = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
            conversational_model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            pipeline = RAGPipeline(web_searcher=web_searcher, model=conversational_model)
            generated_text = retry_sync(retries=3, delay=1)(pipeline.answer)(text)
            audio_path = f"./history_result/output_search_{ts}.mp3"
            retry_sync(retries=3, delay=1)(PollyTTS().synthesize)(generated_text, audio_path)

        elif task_type == "行動":
            action_decomposer = ActionDecomposer()
            generated_text = retry_sync(retries=3, delay=1)(action_decomposer.decompose)(text)

        if generated_text:
            socketio.emit('text_response', generated_text)

        if audio_path and Path(audio_path).exists():
            logger.info(f"[handle_text] 音檔生成完成：{audio_path}")
            with app.app_context():
                server_name = app.config.get('SERVER_NAME', 'localhost:5000')
                if not server_name.startswith('http'):
                    server_name = f"https://{server_name}"

                relative_path = f"/history_result/{os.path.basename(audio_path)}"  # ⭐自己組
                audio_url = f"{server_name}{relative_path}"                       # ⭐直接拼
                # （這時不用再replace了！）

            socketio.emit('expression', '/static/animations/speaking.gif')
            socketio.emit('audio_url', audio_url)



        socketio.emit('status', '✅ 已完成。')

    except asyncio.CancelledError:
        logger.info("[handle_text] 任務被取消")
        raise
    except Exception as e:
        logger.error(f"[handle_text] 發生錯誤：{e}")



async def cancellable_socket_handle_text(text: str):
    global current_task
    with current_task_lock:
        if current_task and not current_task.done():
            logger.info("[cancellable_socket_handle_text] 取消上一個任務...")
            current_task.cancel()

        loop = asyncio.get_running_loop()
        current_task = loop.create_task(handle_text(text))

# --- 路由 ---
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/history_result/<filename>')
def get_audio(filename):
    return send_from_directory('history_result', filename)

# --- 主程式 ---
if __name__ == '__main__':
    os.makedirs('history_result', exist_ok=True)
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
