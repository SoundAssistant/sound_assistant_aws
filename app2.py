import os
import threading
import asyncio
import time
import logging
import base64
import tempfile
import subprocess
import json # <-- 新增
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory, url_for
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
import boto3 # <-- 新增
from botocore.config import Config # <-- 新增

# --- 環境初始化 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
# 如果部署在非本地環境，可能需要移除或修改 SERVER_NAME
# app.config['SERVER_NAME'] = 'localhost:5000' # 可能需要調整或移除
socketio = SocketIO(app, cors_allowed_origins="*")

current_task = None
current_task_lock = threading.Lock()
is_active = False # <-- 新增：系統啟動狀態
is_active_lock = threading.Lock() # <-- 新增：狀態鎖

# --- 啟動時檢查 ffmpeg ---
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
except Exception:
    logger.error("❌ 找不到 ffmpeg，請安裝 ffmpeg。")
    raise

# ---------- Bedrock 參數 ----------
REGION    = "us-west-2"                                   # 改成你的區域 (與 Transcribe 同區)
MODEL_ID  = "anthropic.claude-3-haiku-20240307-v1:0"       # 改成你的模型
BEDROCK   = boto3.client("bedrock-runtime",
                         region_name=REGION,
                         config=Config(read_timeout=300, connect_timeout=10)) # 增加超時時間
# ----------------------------------

# ---------- 分類提示 ----------
_CLASSIFY_PROMPT = """
請判斷下列文字的意圖，只能回答以下四個字串之一：
START/STOP/INTERRUPT/COMMAND
START (啟動關鍵字): 例如「啟動」、「你好」、「哈囉」、「機器人」等等
STOP (結束關鍵字): 例如「關閉」、「再見」、「掰掰」、「結束」等等
INTERRUPT (打斷關鍵字): 例如「等一下」、「暫停」、「閉嘴」、「停」等等
COMMAND (一般命令): 前三者以外，都歸類於此

**務必輸出其中一個字串**

文字：「{text}」
"""
# ----------------------------------

# ---------- Bedrock 分類 ----------
async def classify_intent(text: str) -> str:
    user_prompt = _CLASSIFY_PROMPT.format(text=text.replace('"', '\\"'))
    logger.info(f"[classify_intent] 準備分類文字：{text}")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10, # 稍微增加 token 數以防萬一
        "temperature": 0,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    @retry_sync(retries=2, delay=0.5) # 加入重試
    def _invoke():
        try:
            resp = BEDROCK.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8")
            )
            data = json.loads(resp["body"].read())
            logger.debug(f"[classify_intent] Bedrock 原始回應: {data}") # Debug Log
            content = data.get("content", []) # 確保是 list
            if isinstance(content, list) and content:
                result_text = content[0].get("text", "").strip().upper()
                # 再次檢查是否在允許的意圖內
                valid_intents = {"START", "STOP", "INTERRUPT", "COMMAND"}
                if result_text in valid_intents:
                    logger.info(f"[classify_intent] Bedrock 分類結果：{result_text}")
                    return result_text
                else:
                    logger.warning(f"[classify_intent] Bedrock 回應 '{result_text}' 非預期意圖，歸類為 COMMAND")
                    return "COMMAND" # 如果回傳怪東西，預設為 COMMAND
            else:
                logger.warning(f"[classify_intent] Bedrock 回應格式錯誤或無內容，歸類為 COMMAND")
                return "COMMAND" # 或 IGNORE，視需求調整
        except Exception as e:
            logger.error(f"[classify_intent] Bedrock invoke 失敗：{e}")
            raise # 讓 retry 機制處理

    try:
        # 使用 asyncio.to_thread 執行同步函數
        intent = await asyncio.to_thread(_invoke)
        return intent
    except Exception as e:
        logger.error(f"[classify_intent] Bedrock 分類重試後仍然失敗：{e}")
        return "IGNORE" # 多次失敗後回傳 IGNORE
# -----------------------------------


# --- Transcript Handler ---
class MyTranscriptHandler(TranscriptResultStreamHandler):
    # 不再需要 classify_intent 方法，移到外面作為獨立函數

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                text = result.alternatives[0].transcript.strip()
                if text:
                    logger.info(f"[TranscribeHandler] 轉出文字：{text}")
                    # ⭐ 修改：不再直接呼叫 cancellable_socket_handle_text
                    # 改為呼叫新的意圖處理函數
                    await handle_intent_from_text(text)

# --- HTML 模板 (維持不變) ---
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
      min-height: 30px; /* 避免文字變換時跳動 */
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
  <img id="expression" src="/static/animations/idle.gif" /> <div id="status">⏳ 系統待機中，請說啟動詞...</div> <div id="volume_bar"><div id="volume_fill"></div></div>
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
let currentSystemStatus = "⏳ 系統待機中，請說啟動詞..."; // 新增：追蹤系統狀態

const baseThreshold = 0.08;             // 基本啟動門檻
let dynamicThreshold = baseThreshold;    // 動態啟動門檻
const silenceThreshold = 0.02;           // 判定無聲
const silenceDelay = 1500;               // 錄音中無聲多久停止錄音（毫秒） - 稍微縮短
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
    hasRecordedOnce = true; // 標記已錄音過
    if (audioChunks.length > 0) {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];

      const reader = new FileReader();
      reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];
        // ⭐ 送出前顯示處理中狀態
        status.innerText = '🧠 正在分析語音...';
        expr.src = '/static/animations/thinking.gif';
        socket.emit('audio_blob', base64Audio);
      };
      reader.readAsDataURL(audioBlob);
    } else {
      // 如果沒有錄到聲音，則直接重新監聽
      setTimeout(startListening, 100); // 短暫延遲避免過於頻繁
    }
  });

  startListening(); // 啟動監聽
}

function startListening() {
  isRecording = false;
  recordingStartTime = null;
  silenceStart = null;
  weakNoiseStart = null;
  backgroundVolumes = [];
  audioChunks = [];

  // ⭐ 根據 currentSystemStatus 決定初始動畫和文字
  status.innerText = currentSystemStatus;
  if (currentSystemStatus.includes("待機")) {
     expr.src = '/static/animations/idle.gif';
  } else if (currentSystemStatus.includes("啟動")) {
     expr.src = '/static/animations/wakeup.svg'; // 或 listening.gif
  } else {
     expr.src = '/static/animations/thinking.gif'; // 預設 thinking
  }

  monitorVolume(); // 開始監控音量
}

function monitorVolume() {
  if (!stream || !analyser) return; // 防禦性檢查

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

    if (backgroundVolumes.length > 10) { // 收集足夠樣本後再計算
        const avgBackground = backgroundVolumes.reduce((a, b) => a + b, 0) / backgroundVolumes.length;
        if (avgBackground > 0.05) {
          dynamicThreshold = Math.min(0.15, baseThreshold + (avgBackground - 0.05) * 1.5); // 稍微提高動態門檻影響力
        } else {
          dynamicThreshold = baseThreshold;
        }
    } else {
        dynamicThreshold = baseThreshold; // 樣本不足時用基礎門檻
    }
  }

  // --- 小聲雜訊忽略 ---
  if (!isRecording) {
    if (volume > silenceThreshold && volume < dynamicThreshold) {
      if (!weakNoiseStart) weakNoiseStart = now;
      if (now - weakNoiseStart > weakNoiseIgnoreTime) {
        // console.log('💤 小聲雜訊超過3秒，忽略');
        weakNoiseStart = null;
        backgroundVolumes = []; // 重置背景音量計算
      }
    } else {
      weakNoiseStart = null;
    }
  }

  // --- 錄音邏輯 ---
  if (!isRecording) {
    // 只有音量大於動態門檻，且不是剛忽略的小聲雜訊時才啟動
    if (volume > dynamicThreshold && weakNoiseStart === null) {
      console.log('🎙️ 偵測到說話，開始錄音！');
      try {
        if (mediaRecorder.state === 'inactive') {
            mediaRecorder.start();
            recordingStartTime = now;
            silenceStart = null;
            isRecording = true;
            status.innerText = '🎤 錄音中...';
            expr.src = '/static/animations/listening.gif'; // 使用 listening 動畫
        }
      } catch (e) {
        console.error("無法啟動錄音:", e);
        // 可能需要重新初始化麥克風
        prepareMicrophone();
        return; // 停止這次的 monitor
      }
    }
  } else { // 正在錄音中
    if (volume > silenceThreshold) {
      silenceStart = null; // 有聲音，重置靜音計時器
    } else { // 低於靜音門檻
      if (!silenceStart) silenceStart = now; // 開始計時靜音
      if (now - silenceStart > silenceDelay) {
        console.log(`🛑 靜音超過 ${silenceDelay / 1000} 秒，停止錄音`);
        try {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop(); // 停止錄音會觸發 'stop' 事件
          }
        } catch (e) {
            console.error("無法停止錄音:", e);
            // 即使出錯，也要嘗試回到監聽狀態
            setTimeout(startListening, 100);
        }
        return; // 停止這次的 monitor
      }
    }
    // 檢查是否超過最大錄音時間
    if (now - recordingStartTime > maxRecordingTime) {
      console.log(`⏰ 錄音超過 ${maxRecordingTime / 1000} 秒，強制停止`);
       try {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop(); // 停止錄音會觸發 'stop' 事件
          }
       } catch (e) {
           console.error("無法停止錄音 (超時):", e);
           setTimeout(startListening, 100);
       }
      return; // 停止這次的 monitor
    }
  }

  requestAnimationFrame(monitorVolume); // 持續監控
}

// --- 處理 server 回傳訊息 ---
socket.on('expression', (path) => {
  console.log("Received expression:", path);
  expr.src = path;
});

socket.on('audio_url', (url) => {
  console.log("Received audio URL:", url);
  expr.src = '/static/animations/speaking.gif';
  player.pause();
  player.src = url;
  player.load();
  player.play().catch(err => console.error("❌ 播放失敗", err));
  player.onended = () => {
    // ⭐ 播放完畢後，根據 currentSystemStatus 決定回到哪個狀態
    status.innerText = currentSystemStatus;
    if (currentSystemStatus.includes("待機")) {
       expr.src = '/static/animations/idle.gif';
    } else if (currentSystemStatus.includes("啟動")) {
       expr.src = '/static/animations/wakeup.svg'; // 或 listening.gif
    } else {
       expr.src = '/static/animations/thinking.gif';
    }

    // 刪除播放完的檔案
    if (player.src.includes("/history_result/")) {
      const filename = player.src.split("/history_result/")[1];
      socket.emit('delete_audio', filename);
    }
    // ⭐ 播放完畢後，自動重新開始監聽
    setTimeout(startListening, 500);
  };
  player.onerror = (e) => {
    console.error("音訊播放錯誤:", e);
    // 即使播放錯誤，也要回到監聽狀態
     status.innerText = currentSystemStatus;
     if (currentSystemStatus.includes("待機")) {
       expr.src = '/static/animations/idle.gif';
    } else if (currentSystemStatus.includes("啟動")) {
       expr.src = '/static/animations/wakeup.svg'; // 或 listening.gif
    } else {
       expr.src = '/static/animations/thinking.gif';
    }
    setTimeout(startListening, 500);
  };
});

socket.on('status', (msg) => {
  console.log("Received status:", msg);
  currentSystemStatus = msg; // ⭐ 更新前端追蹤的狀態
  status.innerText = msg;
  // 可以根據狀態文字包含的關鍵字來改變表情，但由後端直接發 expression 事件更可靠
});

socket.on('user_query', (text) => {
  console.log("Received user query:", text);
  latestUserQuery = text;
});

socket.on('text_response', (text) => {
  console.log("Received text response:", text);
  const entry = document.createElement('div');
  entry.className = 'chat_entry';
  entry.innerHTML = `
    <div class="user_query">🧑 ${latestUserQuery || '...'}</div>
    <div class="bot_response">🤖 ${text}</div>
  `;
  chatLog.appendChild(entry);
  // 捲動到底部
  // 使用 setTimeout 確保 DOM 更新完成後再捲動
  setTimeout(() => {
    chatLog.scrollTop = chatLog.scrollHeight;
  }, 0);
  latestUserQuery = null; // 清除上次的 query
});
</script>

</body>
</html>
'''

@socketio.on('delete_audio')
def delete_audio(filename):
    try:
        # 安全地組合路徑
        base_dir = Path('history_result').resolve() # 獲取絕對路徑
        path = base_dir / filename
        # 確保檔案在預期目錄下，防止路徑遍歷攻擊
        if path.is_file() and path.parent == base_dir:
            os.remove(path)
            logger.info(f"[delete_audio] 已刪除檔案：{path}")
        else:
            logger.warning(f"[delete_audio] 嘗試刪除無效路徑或不在允許目錄的檔案：{filename}")
    except Exception as e:
        logger.error(f"[delete_audio] 刪除檔案 '{filename}' 失敗：{e}")

# --- 音訊處理 ---
@socketio.on('audio_blob')
def handle_audio_blob(base64_audio):
    logger.info("[handle_audio_blob] 收到音訊 blob，準備轉換...")
    # 前端已在送出前切換 thinking.gif

    try:
        audio_data = base64.b64decode(base64_audio)

        # 使用 .opus 或 .webm，因為前端錄製的是 webm
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
            logger.info(f"[handle_audio_blob] 音訊暫存於：{tmp_file_path}")

        # 使用 asyncio.run_coroutine_threadsafe 在異步事件循環中執行協程
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(process_audio_file(tmp_file_path), loop)

    except Exception as e:
        logger.error(f"[handle_audio_blob] 音訊處理失敗：{e}")
        socketio.emit('status', '❌ 音訊處理失敗')
        # 可能需要觸發前端重新監聽


async def process_audio_file(file_path):
    global is_active # 讓此函數知道目前的啟動狀態
    input_path = Path(file_path)
    pcm_path = input_path.with_suffix('.wav') # 使用 Path 物件處理路徑轉換
    client = None # 初始化，確保 finally 可以檢查
    stream = None # 初始化

    logger.info(f"[process_audio_file] 開始處理音訊檔案: {input_path}")

    try:
        # === 步驟 1: 使用 FFmpeg 轉換音訊 ===
        logger.info(f"[process_audio_file] 準備將 {input_path} 轉換為 {pcm_path}")
        command = [
            "ffmpeg", "-y",       # 覆蓋輸出文件
            "-i", str(input_path), # 輸入文件
            "-ac", "1",           # 單聲道
            "-ar", "16000",       # 16kHz 採樣率
            "-f", "wav",          # 輸出格式為 WAV (PCM)
            str(pcm_path)      # 輸出文件
        ]
        logger.info(f"[process_audio_file] 執行 FFmpeg 命令: {' '.join(command)}")

        # 使用 asyncio 執行子程序
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate() # 等待命令完成並獲取輸出

        # 檢查 FFmpeg 是否成功執行
        if process.returncode != 0:
            # 解碼 stderr 以便閱讀錯誤訊息
            error_message = stderr.decode(errors='ignore').strip()
            logger.error(f"[process_audio_file] FFmpeg 轉換失敗! Return Code: {process.returncode}")
            logger.error(f"[process_audio_file] FFmpeg Stderr: {error_message}")
            # 向前端報告更具體的錯誤
            socketio.emit('status', f'❌ 音訊轉檔失敗 (FFmpeg Error {process.returncode})')
            # 可以在這裡加更多恢復前端狀態的代碼
            # ... (恢復 status 和 expression 的 emit)
            return # 提前結束函數執行
        else:
            ffmpeg_output = stderr.decode(errors='ignore').strip() # FFmpeg 通常把訊息輸出到 stderr
            logger.info("[process_audio_file] FFmpeg 轉換成功.")
            logger.debug(f"[process_audio_file] FFmpeg Output/Info:\n{ffmpeg_output}") # 打印 ffmpeg 訊息供參考
            if not pcm_path.exists():
                 logger.error(f"[process_audio_file] FFmpeg 聲稱成功，但輸出檔案 {pcm_path} 未找到!")
                 socketio.emit('status', '❌ 音訊轉檔後檔案遺失')
                 return

        # === 步驟 2: 讀取轉換後的 WAV 檔案 ===
        logger.info(f"[process_audio_file] 準備讀取轉換後的 WAV 檔案: {pcm_path}")
        try:
            with open(pcm_path, 'rb') as f:
                pcm_data = f.read()
            logger.info(f"[process_audio_file] 成功讀取 WAV 檔案，大小: {len(pcm_data)} bytes")

            # 檢查檔案大小是否合理 (例如，至少大於 WAV header 的大小)
            if len(pcm_data) < 44: # WAV header 通常是 44 bytes
                logger.warning(f"[process_audio_file] WAV 檔案 {pcm_path} 過小 ({len(pcm_data)} bytes)，可能為空或已損壞。")
                # 你可以選擇是報錯還是嘗試繼續
                socketio.emit('status', '🟡 音訊檔案似乎是空的')
                # ... (恢復 status 和 expression 的 emit)
                # return # 如果確定空檔案無法處理，可以取消註解這行
        except Exception as read_err:
            logger.error(f"[process_audio_file] 讀取 WAV 檔案 {pcm_path} 時發生錯誤: {read_err}", exc_info=True)
            socketio.emit('status', '❌ 無法讀取轉換後的音訊')
            return

        # === 步驟 3: 初始化並連接 AWS Transcribe ===
        logger.info(f"[process_audio_file] 準備初始化 TranscribeStreamingClient (Region: {REGION})")
        try:
            client = TranscribeStreamingClient(region=REGION) # 使用全局 REGION
            logger.info("[process_audio_file] TranscribeStreamingClient 初始化成功")

            logger.info("[process_audio_file] 準備啟動 Transcribe Stream...")
            stream = await client.start_stream_transcription(
                language_code="zh-TW",        # 語言代碼
                media_sample_rate_hz=16000,   # 採樣率 (與 ffmpeg 轉換一致)
                media_encoding="pcm",         # 編碼 (與 ffmpeg 轉換一致)
            )
            logger.info("[process_audio_file] Transcribe Stream 啟動成功")
        except Exception as transcribe_init_err:
            logger.error(f"[process_audio_file] 初始化或啟動 Transcribe Stream 時失敗: {transcribe_init_err}", exc_info=True)
            # 可能的原因：AWS憑證、權限、網路、區域錯誤
            socketio.emit('status', '❌ 連接語音辨識服務失敗')
            return

        # === 步驟 4: 異步發送音訊數據和接收結果 ===
        logger.info("[process_audio_file] 準備並發執行 write_chunks 和 read_results")

        async def write_chunks():
            chunk_size = 8000 # 每次發送的字節數，可以調整
            total_sent = 0
            logger.info(f"[write_chunks] 開始發送音訊數據 (總大小: {len(pcm_data)} bytes, 塊大小: {chunk_size})")
            try:
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i+chunk_size]
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    total_sent += len(chunk)
                    # logger.debug(f"[write_chunks] 已發送 chunk {i//chunk_size + 1}, 大小: {len(chunk)}")
                    await asyncio.sleep(0.1) # 短暫等待，避免發送過快導致問題
                logger.info(f"[write_chunks] 所有音訊數據發送完畢 (共 {total_sent} bytes)")
                await stream.input_stream.end_stream()
                logger.info("[write_chunks] 已發送 end_stream 信號")
            except Exception as write_err:
                logger.error(f"[write_chunks] 發送音訊串流時發生錯誤: {write_err}", exc_info=True)
                # 即使出錯，也嘗試關閉流 (儘管可能已經關閉)
                try: await stream.input_stream.end_stream()
                except: pass
                raise # 將錯誤向上拋出，讓 gather 知道

        async def read_results():
            logger.info("[read_results] 開始處理 Transcribe 返回的事件")
            # ⭐ 注意：MyTranscriptHandler 實例應該在這裡創建
            handler = MyTranscriptHandler(stream.output_stream)
            try:
                await handler.handle_events() # 這個方法會持續處理直到 stream 結束
                logger.info("[read_results] Transcribe 事件處理循環正常結束")
            except Exception as read_err:
                logger.error(f"[read_results] 處理 Transcribe 事件時發生錯誤: {read_err}", exc_info=True)
                raise # 將錯誤向上拋出

        try:
            # 使用 asyncio.gather 並發執行寫入和讀取
            await asyncio.gather(write_chunks(), read_results())
            logger.info("[process_audio_file] Transcribe 串流處理成功完成")
        except Exception as gather_err:
            # gather 會在任何一個任務出錯時停止並拋出錯誤
            logger.error(f"[process_audio_file] Transcribe 串流處理期間發生錯誤 (來自 gather): {gather_err}", exc_info=True)
            socketio.emit('status', '❌ 語音辨識過程中斷')
            # 這裡不需要 return，因為 finally 會執行清理

    except Exception as overall_err:
        # 捕獲上面步驟中未被特定 try-except 捕獲的任何其他錯誤
        logger.error(f"[process_audio_file] 處理音訊時發生未預期的錯誤: {overall_err}", exc_info=True)
        socketio.emit('status', f'❌ 音訊處理時發生未知錯誤')
        # 可以在這裡加入恢復前端狀態的邏輯
        current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
        socketio.emit('status', current_status)
        expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
        socketio.emit('expression', expression)

    finally:
        # === 步驟 5: 清理資源 ===
        logger.info("[process_audio_file] 執行 finally 區塊進行清理")
        # 安全地關閉 Transcribe stream (如果存在且未關閉)
        # 注意：Transcribe Streaming SDK 可能沒有顯式的 close() 方法，
        # end_stream() 應該是主要的關閉信號。
        # if stream and stream.input_stream and not stream.input_stream.is_closed():
        #     try:
        #         logger.info("[process_audio_file] 嘗試在 finally 中關閉 input_stream")
        #         await stream.input_stream.end_stream()
        #     except Exception as close_err:
        #         logger.warning(f"[process_audio_file] 在 finally 中關閉 Transcribe stream 時出錯: {close_err}")

        # 清理臨時檔案
        logger.info("[process_audio_file] 準備刪除臨時檔案")
        files_to_delete = [input_path, pcm_path]
        for file_to_delete in files_to_delete:
            try:
                if file_to_delete and file_to_delete.exists():
                    os.remove(file_to_delete)
                    logger.info(f"[process_audio_file] 已成功刪除檔案: {file_to_delete}")
                # else:
                #     logger.debug(f"[process_audio_file] 檔案不存在或路徑為空，無需刪除: {file_to_delete}")
            except OSError as delete_err:
                # 如果刪除失敗，記錄警告但不要讓它中斷程式
                logger.warning(f"[process_audio_file] 刪除臨時檔案 {file_to_delete} 時失敗: {delete_err}")
            except Exception as e:
                logger.warning(f"[process_audio_file] 刪除檔案 {file_to_delete} 時發生未知錯誤: {e}")
        logger.info("[process_audio_file] 清理完成")
        

		
# --- 新增：根據文字處理意圖 ---
async def handle_intent_from_text(text: str):
    global is_active, current_task

    socketio.emit('status', f"💬 收到: \"{text}\"，分析意圖...")
    socketio.emit('expression', '/static/animations/thinking.gif') # 分析意圖時也用 thinking
    socketio.emit('user_query', text) # 先顯示用戶說的話

    intent = await classify_intent(text)
    logger.info(f"[handle_intent_from_text] 文字: '{text}', 意圖: {intent}")

    with is_active_lock: # 確保狀態修改的原子性
        if intent == "START":
            if not is_active:
                is_active = True
                logger.info("[handle_intent_from_text] 系統啟動")
                # 取消可能存在的舊任務 (雖然理論上 inactive 時不該有)
                with current_task_lock:
                    if current_task and not current_task.done():
                        logger.info("[handle_intent_from_text] (START) 取消舊任務...")
                        current_task.cancel()
                        current_task = None
                socketio.emit('status', '🟢 已啟動，請說指令...')
                socketio.emit('expression', '/static/animations/wakeup.svg') # 啟動動畫
            else:
                logger.info("[handle_intent_from_text] 系統已啟動，忽略 START 指令")
                socketio.emit('status', '🟢 已啟動，請說指令...') # 維持啟動狀態提示
                socketio.emit('expression', '/static/animations/wakeup.svg')

        elif intent == "STOP":
            if is_active:
                is_active = False
                logger.info("[handle_intent_from_text] 系統關閉")
                with current_task_lock:
                    if current_task and not current_task.done():
                        logger.info("[handle_intent_from_text] (STOP) 取消進行中任務...")
                        current_task.cancel()
                        current_task = None
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...')
                socketio.emit('expression', '/static/animations/idle.gif') # 待機動畫
            else:
                logger.info("[handle_intent_from_text] 系統已關閉，忽略 STOP 指令")
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...') # 維持待機狀態提示
                socketio.emit('expression', '/static/animations/idle.gif')

        elif intent == "INTERRUPT":
            if is_active:
                logger.info("[handle_intent_from_text] 收到打斷指令")
                with current_task_lock:
                    if current_task and not current_task.done():
                        logger.info("[handle_intent_from_text] (INTERRUPT) 取消進行中任務...")
                        current_task.cancel()
                        current_task = None
                        socketio.emit('status', '🟡 已中斷，請說新指令...')
                        socketio.emit('expression', '/static/animations/listening.gif') # 或 thinking
                    else:
                        logger.info("[handle_intent_from_text] (INTERRUPT) 無任務可中斷，等待新指令...")
                        socketio.emit('status', '🟢 已啟動，請說指令...') # 回到等待指令狀態
                        socketio.emit('expression', '/static/animations/wakeup.svg')
            else:
                logger.info("[handle_intent_from_text] 系統未啟動，忽略 INTERRUPT 指令")
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...') # 維持待機狀態提示
                socketio.emit('expression', '/static/animations/idle.gif')

        elif intent == "COMMAND":
            if is_active:
                logger.info("[handle_intent_from_text] 執行指令型任務...")
                socketio.emit('status', f'🚀 收到指令: "{text}"，執行中...')
                socketio.emit('expression', '/static/animations/thinking.gif')
                # 執行原本的任務處理邏輯
                await cancellable_socket_handle_text(text)
            else:
                logger.info("[handle_intent_from_text] 系統未啟動，忽略 COMMAND 指令")
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...')
                socketio.emit('expression', '/static/animations/idle.gif')

        elif intent == "IGNORE":
            logger.info("[handle_intent_from_text] 忽略無法分類或無效的指令")
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status) # 回復當前狀態提示
            expression = '/static/animations/wakeup.svg' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression) # 回復對應表情

        # ⭐ 無論如何，最後都讓前端重新監聽 (除非是 STOP)
        #    (這部分邏輯移到前端的 audio_url onended 和 onerror 處理)
        # if intent != "STOP":
        #     socketio.emit('reset_listening') # 通知前端可以開始下一次監聽

# --- 任務處理 (handle_text 函數基本不變) ---
async def handle_text(text: str):
    global is_active # 需要知道狀態，雖然主要邏輯在外層處理了
    try:
        # 這個函數現在只處理確認過的 COMMAND
        logger.info(f"[handle_text] 開始處理命令：{text}")
        # socketio.emit('status', f"🚀 執行中：{text}") # 狀態已在 handle_intent_from_text 更新
        # socketio.emit('user_query', text) # 已在 handle_intent_from_text 發送

        # --- 任務分類 (可選，如果 Bedrock 分類已足夠，這裡可以簡化) ---
        # 如果希望保留原有的 TaskClassifier，可以繼續使用
        task_classifier = TaskClassifier()
        # 注意 retry_sync 是同步的，在 async 函數中使用需要 to_thread
        # task_type, _ = await asyncio.to_thread(retry_sync(retries=3, delay=1)(task_classifier.classify_task), text)
        # 或者，簡化處理，直接假設是聊天或查詢 (如果 Bedrock 分類夠準)
        # 這裡我們假設仍然使用 TaskClassifier
        classify_func = retry_sync(retries=3, delay=1)(task_classifier.classify_task)
        task_type, _ = await asyncio.to_thread(classify_func, text)

        logger.info(f"[handle_text] 任務分類結果：{task_type}")

        # socketio.emit('expression', '/static/animations/thinking.gif') # 已在外層設定

        audio_path = None
        generated_text = None
        ts = time.strftime('%Y%m%d_%H%M%S')
        output_dir = Path("./history_result")
        output_dir.mkdir(exist_ok=True) # 確保目錄存在

        # 使用 await asyncio.to_thread 來執行同步的 retry_sync 包裹的函數
        if task_type == "聊天":
            chat_model = Chatbot(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            chat_func = retry_sync(retries=3, delay=1)(chat_model.chat)
            generated_text = await asyncio.to_thread(chat_func, text)
            if generated_text:
                audio_path = output_dir / f"output_chat_{ts}.mp3"
                tts_func = retry_sync(retries=3, delay=1)(PollyTTS().synthesize)
                await asyncio.to_thread(tts_func, generated_text, str(audio_path))

        elif task_type == "查詢":
            web_searcher = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
            conversational_model = ConversationalModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
            pipeline = RAGPipeline(web_searcher=web_searcher, model=conversational_model)
            answer_func = retry_sync(retries=3, delay=1)(pipeline.answer)
            generated_text = await asyncio.to_thread(answer_func, text)
            if generated_text:
                audio_path = output_dir / f"output_search_{ts}.mp3"
                tts_func = retry_sync(retries=3, delay=1)(PollyTTS().synthesize)
                await asyncio.to_thread(tts_func, generated_text, str(audio_path))

        elif task_type == "行動":
            action_decomposer = ActionDecomposer()
            decompose_func = retry_sync(retries=3, delay=1)(action_decomposer.decompose)
            generated_text = await asyncio.to_thread(decompose_func, text)
            # 行動類型通常沒有語音回覆，但可以根據需求添加
            if generated_text:
                 logger.info(f"[handle_text] 行動分解結果: {generated_text}")
                 # 可以選擇性地念出分解結果
                 # audio_path = output_dir / f"output_action_{ts}.mp3"
                 # tts_func = retry_sync(retries=3, delay=1)(PollyTTS().synthesize)
                 # await asyncio.to_thread(tts_func, f"好的，收到行動指令。 {generated_text}", str(audio_path)) # 例如

        else:
            # 未知任務類型，可以給一個通用回覆
             logger.warning(f"[handle_text] 未知的任務類型: {task_type}")
             generated_text = "抱歉，我不太理解這個指令。"
             audio_path = output_dir / f"output_unknown_{ts}.mp3"
             tts_func = retry_sync(retries=3, delay=1)(PollyTTS().synthesize)
             await asyncio.to_thread(tts_func, generated_text, str(audio_path))


        if generated_text:
            socketio.emit('text_response', generated_text) # 發送文字回覆

        if audio_path and audio_path.exists():
            logger.info(f"[handle_text] 音檔生成完成：{audio_path}")
            # 使用 with app.app_context() 來獲取 url_for
            with app.app_context():
                # 使用 Path 物件獲取檔名
                audio_url = url_for('get_audio', filename=audio_path.name, _external=False) # 使用相對路徑
                logger.info(f"[handle_text] 生成 Audio URL: {audio_url}")
            socketio.emit('expression', '/static/animations/speaking.gif')
            socketio.emit('audio_url', audio_url) # 發送音檔 URL 給前端播放
        else:
            # 如果沒有音檔生成（例如只有文字回覆或行動指令），也要恢復狀態
            logger.info("[handle_text] 無音檔生成，任務處理完畢")
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/wakeup.svg' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression)
            # ⭐ 同樣，讓前端在適當時候 (例如 text_response 收到後) 重新監聽

        # socketio.emit('status', '✅ 已完成。') # 狀態由 audio_url 的 onended 或 text_response 處理更佳

    except asyncio.CancelledError:
        logger.info("[handle_text] 任務被取消")
        socketio.emit('status', '🟡 任務已中斷')
        socketio.emit('expression', '/static/animations/listening.gif') # 或 idle/wakeup
        raise # 重新拋出異常，讓上層知道被取消了
    except Exception as e:
        logger.error(f"[handle_text] 處理命令時發生錯誤：{e}", exc_info=True) # 打印詳細錯誤
        socketio.emit('status', f'❌ 執行命令時出錯: {e}')
        # 出錯後恢復狀態
        current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
        socketio.emit('status', current_status)
        expression = '/static/animations/wakeup.svg' if is_active else '/static/animations/idle.gif'
        socketio.emit('expression', expression)


async def cancellable_socket_handle_text(text: str):
    global current_task
    with current_task_lock:
        if current_task and not current_task.done():
            logger.info("[cancellable_socket_handle_text] 收到新命令，取消上一個任務...")
            current_task.cancel()
            # 等待上一個任務確實被取消 (可選，但有助於資源釋放)
            # try:
            #     await asyncio.wait_for(current_task, timeout=1.0)
            # except (asyncio.CancelledError, asyncio.TimeoutError):
            #     pass # 忽略取消或超時錯誤

        loop = asyncio.get_running_loop()
        logger.info(f"[cancellable_socket_handle_text] 創建新任務來處理命令: {text}")
        current_task = loop.create_task(handle_text(text))
        # 可以在任務完成時添加回調，用於清理或記錄
        # current_task.add_done_callback(lambda t: logger.info(f"任務 {t.get_name()} 完成"))


# --- 路由 ---
@app.route('/')
def index():
    # 確保在請求上下文中渲染模板
    return render_template_string(HTML)

@app.route('/history_result/<path:filename>') # 使用 path converter 處理可能包含子目錄的檔名
def get_audio(filename):
    logger.debug(f"請求音檔: {filename}")
    # 使用 safe_join 確保安全，並從絕對路徑提供服務
    directory = Path('history_result').resolve()
    try:
        # 檢查請求的檔案是否在允許的目錄下
        requested_path = (directory / filename).resolve()
        if requested_path.is_file() and requested_path.parent == directory:
             logger.debug(f"從目錄 {directory} 提供檔案 {filename}")
             return send_from_directory(directory, filename)
        else:
             logger.warning(f"拒絕存取不在允許目錄的檔案: {filename}")
             return "Forbidden", 403
    except FileNotFoundError:
        logger.error(f"請求的音檔不存在: {filename}")
        return "Not Found", 404
    except Exception as e:
        logger.error(f"提供音檔時發生錯誤 ({filename}): {e}")
        return "Internal Server Error", 500

# --- 主程式 ---
if __name__ == '__main__':
    # 確保 history_result 目錄存在
    os.makedirs('history_result', exist_ok=True)
    logger.info("啟動 SocketIO Server on 0.0.0.0:5000")
    # 建議不要在生產環境中使用 allow_unsafe_werkzeug=True
    # 考慮使用更健壯的 WSGI 伺服器如 gunicorn 或 uvicorn 配合 eventlet 或 gevent
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
    # debug=True, use_reloader=False 可以在開發時幫助調試，但生產環境應設為 False
    # use_reloader=False 避免重啟時 asyncio 事件循環出問題
