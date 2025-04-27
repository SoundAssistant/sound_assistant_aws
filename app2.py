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
# from urllib.parse import urljoin # <-- 不再需要
# from flask import request # <-- 如果沒有其他地方用到，可以移除
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
# SERVER_NAME 主要影響 url_for(_external=True)。如果只用相對路徑，可以不設定或謹慎設定。
# ngrok 的免費 URL 可能會變動。
# app.config['SERVER_NAME'] = '0747-34-222-37-198.ngrok-free.app' # 保留，但 url_for 會用相對路徑
app.config['PREFERRED_URL_SCHEME'] = 'https' # 告知 Flask 使用 https 生成外部 URL (如果需要的話)

socketio = SocketIO(app, cors_allowed_origins="*")

current_task = None
current_task_lock = threading.Lock()
is_active = False # <-- 新增：系統啟動狀態
is_active_lock = threading.Lock() # <-- 新增：狀態鎖

# ---------- Bedrock 參數 ----------
# 確保 AWS 憑證已配置 (環境變數、IAM Role、~/.aws/credentials 等)
REGION    = "us-west-2" # <--- 確認這是你的 AWS 區域
MODEL_ID  = "anthropic.claude-3-haiku-20240307-v1:0" # <--- 確認模型 ID 和權限
BEDROCK_CONFIG = Config(
    region_name=REGION,
    connect_timeout=10, # 連線超時 (秒)
    read_timeout=300    # 讀取超時 (秒) - 分類可能很快，但保留餘裕
)
BEDROCK   = boto3.client("bedrock-runtime", config=BEDROCK_CONFIG)
# ----------------------------------

# ---------- 分類提示 ----------
# (與上次相同)
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
# (與上次相同，包含重試和錯誤處理)
async def classify_intent(text: str) -> str:
    user_prompt = _CLASSIFY_PROMPT.format(text=text.replace('"', '\\"'))
    logger.info(f"[classify_intent] 準備分類文字：{text}")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    # 為了在 asyncio 中執行同步的 retry_sync，我們定義一個內部同步函數
    @retry_sync(retries=2, delay=0.5)
    def _invoke_sync():
        try:
            resp = BEDROCK.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8")
            )
            data = json.loads(resp["body"].read())
            logger.debug(f"[classify_intent] Bedrock 原始回應: {data}")
            content = data.get("content", [])
            if isinstance(content, list) and content:
                result_text = content[0].get("text", "").strip().upper()
                valid_intents = {"START", "STOP", "INTERRUPT", "COMMAND"}
                if result_text in valid_intents:
                    logger.info(f"[classify_intent] Bedrock 分類結果：{result_text}")
                    return result_text
                else:
                    logger.warning(f"[classify_intent] Bedrock 回應 '{result_text}' 非預期意圖，歸類為 COMMAND")
                    return "COMMAND"
            else:
                logger.warning(f"[classify_intent] Bedrock 回應格式錯誤或無內容，歸類為 COMMAND")
                return "COMMAND"
        except Exception as e:
            logger.error(f"[classify_intent] Bedrock invoke 失敗：{e}")
            raise # 讓 retry 機制處理

    try:
        # 使用 asyncio.to_thread 執行同步的、包含重試的內部函數
        intent = await asyncio.to_thread(_invoke_sync)
        return intent
    except Exception as e:
        logger.error(f"[classify_intent] Bedrock 分類重試後仍然失敗：{e}")
        return "IGNORE"
# -----------------------------------


# --- 啟動時檢查 ffmpeg ---
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    logger.info("✅ ffmpeg 檢查成功")
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
                    logger.info(f"[TranscribeHandler] 轉出文字：{text}")
                    # ⭐ 修改：調用新的意圖處理函數
                    await handle_intent_from_text(text)

# --- HTML 模板 (套用上次的修改，包含狀態處理) ---
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
// ⭐ 獲取當前的 base URL，用於 Socket.IO 連線 (如果需要跨域)
// const currentHost = window.location.origin;
// const socket = io(currentHost); // 連接到當前服務器

// ⭐ 如果 Socket.IO Server 和 Web Server 在同一個 origin，可以直接用 io()
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
let hasRecordedOnce = false; // 保持這個變數，用於判斷是否是首次啟動
let currentSystemStatus = "⏳ 系統待機中，請說啟動詞..."; // 新增：追蹤系統狀態

const baseThreshold = 0.08;             // 基本啟動門檻
let dynamicThreshold = baseThreshold;    // 動態啟動門檻
const silenceThreshold = 0.02;           // 判定無聲
const silenceDelay = 1500;               // 錄音中無聲多久停止錄音（毫秒） - 沿用上次調整
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
      // 可以嘗試提示用戶檢查麥克風權限
    }
  });
};

async function prepareMicrophone() {
  if (stream) { // 如果已有 stream，先停止舊的
      stream.getTracks().forEach(track => track.stop());
      stream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
      await audioContext.close();
      audioContext = null;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    mediaRecorder.addEventListener('dataavailable', event => {
      if (event.data.size > 0) { // 確保有數據
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener('stop', async () => {
      // hasRecordedOnce = true; // 這個標記似乎沒在 startListening 用到，可以考慮移除或修改邏輯
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // 瀏覽器通常錄製 webm 或 ogg
        audioChunks = []; // 清空 chunks

        const reader = new FileReader();
        reader.onloadend = () => {
          const base64Audio = reader.result.split(',')[1];
          status.innerText = '🧠 正在分析語音...';
          expr.src = '/static/animations/thinking.gif';
          socket.emit('audio_blob', base64Audio);
          console.log("Audio blob sent to server.");
        };
        reader.readAsDataURL(audioBlob);
      } else {
        console.log("No audio chunks recorded, restarting listening.");
        // 如果沒有錄到聲音，則直接重新監聽
        setTimeout(startListening, 100); // 短暫延遲
      }
    });

    startListening(); // 啟動監聽
    console.log("Microphone prepared and listening started.");
  } catch (err) {
     console.error('Error preparing microphone:', err);
     status.innerText = '❌ 麥克風啟動失敗，請檢查權限';
     throw err; // 將錯誤拋出，讓 click handler 知道失敗了
  }
}

function startListening() {
  isRecording = false;
  recordingStartTime = null;
  silenceStart = null;
  weakNoiseStart = null;
  backgroundVolumes = [];
  audioChunks = []; // 確保每次重新監聽都清空

  // 根據 currentSystemStatus 決定初始動畫和文字
  status.innerText = currentSystemStatus;
  if (currentSystemStatus.includes("待機")) {
     expr.src = '/static/animations/idle.gif';
  } else if (currentSystemStatus.includes("啟動") || currentSystemStatus.includes("中斷") || currentSystemStatus.includes("指令")) {
     // 啟動、中斷、收到指令後，都顯示等待指令的狀態
     expr.src = '/static/animations/listening.gif'; // 改用 listening
  } else {
     expr.src = '/static/animations/thinking.gif'; // 預設 thinking (例如分析中)
  }
  console.log("startListening called. Status:", currentSystemStatus, "Expression:", expr.src);

  monitorVolume(); // 開始監控音量
}

function monitorVolume() {
  if (!stream || !analyser || !audioContext || audioContext.state === 'closed') {
    console.warn("monitorVolume: Stream or analyser not ready or context closed.");
    // 可以在這裡嘗試重新初始化麥克風或停止監控
    // requestAnimationFrame(prepareMicrophone); // 嘗試重新準備
    return;
  }
   if (mediaRecorder.state === 'recording' && isRecording === false) {
       console.warn("monitorVolume: state mismatch (recording but isRecording=false), fixing state.");
       isRecording = true; // 校正狀態
   }
   if (mediaRecorder.state === 'inactive' && isRecording === true) {
        console.warn("monitorVolume: state mismatch (inactive but isRecording=true), fixing state.");
        isRecording = false; // 校正狀態
   }


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

    if (backgroundVolumes.length > 10) {
        const avgBackground = backgroundVolumes.reduce((a, b) => a + b, 0) / backgroundVolumes.length;
        // 稍微降低動態門檻的敏感度，避免環境噪音誤觸
        dynamicThreshold = Math.max(baseThreshold, Math.min(0.15, baseThreshold + (avgBackground - 0.05) * 1.2));
    } else {
        dynamicThreshold = baseThreshold;
    }
  }

  // --- 小聲雜訊忽略 ---
  if (!isRecording) {
    if (volume > silenceThreshold && volume < dynamicThreshold) {
      if (!weakNoiseStart) weakNoiseStart = now;
      if (now - weakNoiseStart > weakNoiseIgnoreTime) {
        // console.log('💤 小聲雜訊超過3秒，忽略');
        weakNoiseStart = null;
        backgroundVolumes = [];
      }
    } else {
      weakNoiseStart = null;
    }
  }

  // --- 錄音邏輯 ---
  if (!isRecording) {
    // 只有音量大於動態門檻，且不是剛忽略的小聲雜訊時才啟動
    if (volume > dynamicThreshold && weakNoiseStart === null) {
      console.log(`🎙️ Volume (${volume.toFixed(3)}) > Threshold (${dynamicThreshold.toFixed(3)}), Start Recording!`);
      try {
        // 確保 MediaRecorder 處於非錄製狀態
        if (mediaRecorder.state === 'inactive') {
            mediaRecorder.start(); // 開始錄音，會觸發 dataavailable 事件
            recordingStartTime = now;
            silenceStart = null;
            isRecording = true;
            status.innerText = '🎤 錄音中...';
            expr.src = '/static/animations/listening.gif';
        } else {
             console.warn("Attempted to start recording, but state was:", mediaRecorder.state);
        }
      } catch (e) {
        console.error("Error starting mediaRecorder:", e);
        // 嘗試恢復
        prepareMicrophone(); // 重新初始化麥克風
        return;
      }
    }
  } else { // 正在錄音中
    if (volume > silenceThreshold) {
      silenceStart = null; // 有聲音，重置靜音計時器
    } else { // 低於靜音門檻
      if (!silenceStart) silenceStart = now; // 開始計時靜音
      if (now - silenceStart > silenceDelay) {
        console.log(`🛑 Silence detected for > ${silenceDelay / 1000}s, Stopping recording.`);
        try {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop(); // 停止錄音會觸發 'stop' 事件
            // isRecording 會在 'stop' 事件處理函數中被設為 false (間接通過 startListening)
          } else {
               console.warn("Attempted to stop recording due to silence, but state was:", mediaRecorder.state);
               // 如果狀態不對，強制回到監聽狀態
               isRecording = false;
               setTimeout(startListening, 100);
          }
        } catch (e) {
            console.error("Error stopping mediaRecorder (silence):", e);
            isRecording = false; // 確保狀態重置
            setTimeout(startListening, 100); // 嘗試恢復
        }
        return; // 停止這次的 monitor
      }
    }
    // 檢查是否超過最大錄音時間
    if (now - recordingStartTime > maxRecordingTime) {
      console.log(`⏰ Max recording time (${maxRecordingTime / 1000}s) exceeded, Stopping recording.`);
       try {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          } else {
              console.warn("Attempted to stop recording due to max time, but state was:", mediaRecorder.state);
              isRecording = false;
              setTimeout(startListening, 100);
          }
       } catch (e) {
           console.error("Error stopping mediaRecorder (max time):", e);
           isRecording = false;
           setTimeout(startListening, 100);
       }
      return; // 停止這次的 monitor
    }
  }

  // 只要 audioContext 存在且未關閉，就繼續監控
  if (audioContext && audioContext.state === 'running') {
      requestAnimationFrame(monitorVolume);
  } else {
      console.log("AudioContext closed or not running, stopping monitoring.");
  }
}

// --- 處理 server 回傳訊息 ---
socket.on('expression', (path) => {
  console.log("Received expression:", path);
  // 確保路徑是相對的或完整的 URL
  if (path && typeof path === 'string') {
    expr.src = path.startsWith('/') ? path : '/' + path; // 確保有斜線開頭
  }
});

socket.on('audio_url', (url) => {
  console.log("Received audio URL:", url);
  if (!url || typeof url !== 'string') {
      console.error("Invalid audio URL received:", url);
      // 回到監聽狀態
      setTimeout(startListening, 500);
      return;
  }
  expr.src = '/static/animations/speaking.gif';
  player.pause();
  // ⭐ 直接使用後端傳來的相對 URL
  player.src = url;
  player.load();
  player.play().then(() => {
      console.log("Audio playback started:", url);
  }).catch(err => {
      console.error("❌ Audio playback failed:", err);
      // 即使播放失敗，也要嘗試回到監聽狀態
      setTimeout(startListening, 500);
  });

  player.onended = () => {
    console.log("Audio playback finished.");
    // 播放完畢後，根據 currentSystemStatus 決定回到哪個狀態 (由 startListening 處理)
    // ⭐ 觸發 startListening 來恢復介面和監聽
    setTimeout(startListening, 500); // 給一點緩衝時間

    // 刪除播放完的檔案 (檔名從相對路徑中提取)
    if (player.src) { // 確保 src 存在
        try {
            const urlParts = player.src.split('/');
            const filename = urlParts[urlParts.length - 1];
            if (filename && player.src.includes("/history_result/")) {
                 console.log("Requesting deletion of audio:", filename);
                 socket.emit('delete_audio', filename);
            }
        } catch (e) {
             console.error("Error extracting filename for deletion:", e);
        }
    }
  };

  player.onerror = (e) => {
    console.error("Audio player error:", e);
    // 出錯也要回到監聽狀態
    setTimeout(startListening, 500);
  };
});

socket.on('status', (msg) => {
  console.log("Received status:", msg);
  currentSystemStatus = msg; // 更新前端追蹤的狀態
  status.innerText = msg;
  // 表情由 'expression' 事件控制，這裡只更新文字
});

socket.on('user_query', (text) => {
  console.log("Received user query:", text);
  latestUserQuery = text;
});

socket.on('text_response', (text) => {
  console.log("Received text response:", text);
  const entry = document.createElement('div');
  entry.className = 'chat_entry';
  // 防範 XSS，雖然這裡是內部應用，但好習慣是需要的
  const userDiv = document.createElement('div');
  userDiv.className = 'user_query';
  userDiv.textContent = `🧑 ${latestUserQuery || '...'}`; // 使用 textContent
  const botDiv = document.createElement('div');
  botDiv.className = 'bot_response';
  botDiv.textContent = `🤖 ${text}`; // 使用 textContent
  entry.appendChild(userDiv);
  entry.appendChild(botDiv);

  chatLog.appendChild(entry);
  // 捲動到底部
  setTimeout(() => {
    chatLog.scrollTop = chatLog.scrollHeight;
  }, 0);
  latestUserQuery = null; // 清除上次的 query
  // ⭐ 收到文字回應後，也可以觸發 startListening，確保介面恢復
  //    但通常 audio_url 播放完畢觸發更合適
  // setTimeout(startListening, 500);
});

// Socket.IO 連線/斷線處理 (可選)
socket.on('connect', () => {
    console.log('Socket.IO connected:', socket.id);
    // 連線成功後可以做一些初始化，例如請求當前狀態
    // socket.emit('request_status');
});

socket.on('disconnect', (reason) => {
    console.log('Socket.IO disconnected:', reason);
    status.innerText = '❌ 連線中斷，請重新整理';
    expr.src = '/static/animations/idle.gif'; // 或錯誤圖示
    // 停止錄音等相關操作
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
     if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
        audioContext = null;
    }
});

socket.on('connect_error', (error) => {
  console.error('Socket.IO connection error:', error);
  status.innerText = '❌ 無法連線伺服器';
  expr.src = '/static/animations/idle.gif';
});

</script>

</body>
</html>
'''

@socketio.on('delete_audio')
def delete_audio(filename):
    # (沿用上次的安全刪除邏輯)
    try:
        base_dir = Path('history_result').resolve()
        path = (base_dir / filename).resolve() # 使用 resolve 確保路徑是標準化的
        # 再次確認檔案存在且在預期目錄下
        if path.is_file() and path.parent == base_dir:
            os.remove(path)
            logger.info(f"[delete_audio] 已刪除檔案：{path}")
        else:
            logger.warning(f"[delete_audio] 嘗試刪除無效路徑或不在允許目錄的檔案：{filename}")
    except Exception as e:
        logger.error(f"[delete_audio] 刪除檔案 '{filename}' 失敗：{e}")

# --- 音訊處理 ---
# --- 音訊處理 ---
@socketio.on('audio_blob')
def handle_audio_blob(base64_audio):
    # 增加日誌記錄，標識每次調用
    request_id = f"req_{time.monotonic_ns()}" # 創建一個簡單的請求 ID
    logger.info(f"[{request_id}][handle_audio_blob] 收到 audio_blob 事件")

    # 檢查收到的數據類型和初步內容
    if not isinstance(base64_audio, str):
        logger.error(f"[{request_id}][handle_audio_blob] 錯誤：收到的 base64_audio 不是字串，類型為 {type(base64_audio)}")
        socketio.emit('status', '❌ 錯誤：音訊數據格式不對')
        return
    logger.info(f"[{request_id}][handle_audio_blob] 收到 Base64 字串，前 50 字元: {base64_audio[:50]}...")
    logger.info(f"[{request_id}][handle_audio_blob] Base64 字串總長度: {len(base64_audio)}")

    # 前端已切換 thinking.gif

    tmp_file_path = None # 初始化確保 finally 可以檢查

    try:
        # === 步驟 1: Base64 解碼 ===
        logger.info(f"[{request_id}][handle_audio_blob] 嘗試 Base64 解碼...")
        try:
            audio_data = base64.b64decode(base64_audio)
            logger.info(f"[{request_id}][handle_audio_blob] Base64 解碼成功，得到 {len(audio_data)} bytes 的音訊數據")
        except base64.binascii.Error as b64e:
            logger.error(f"[{request_id}][handle_audio_blob] Base64 解碼失敗: {b64e}")
            socketio.emit('status', '❌ 無效的音訊數據 (Base64)')
            return # 解碼失敗，無法繼續
        except Exception as decode_e:
            logger.error(f"[{request_id}][handle_audio_blob] Base64 解碼時發生未知錯誤: {decode_e}", exc_info=True)
            socketio.emit('status', '❌ 音訊數據解碼錯誤')
            return

        # === 步驟 2: 創建並寫入臨時檔案 ===
        # 確保目標目錄存在且可寫
        temp_dir = Path("./temp_audio") # 建議使用一個專用臨時目錄
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{request_id}][handle_audio_blob] 確保臨時目錄存在: {temp_dir.resolve()}")
        except OSError as dir_err:
             logger.error(f"[{request_id}][handle_audio_blob] 無法創建或訪問臨時目錄 {temp_dir.resolve()}: {dir_err}")
             socketio.emit('status', '❌ 伺服器檔案系統錯誤 (Dir)')
             return

        logger.info(f"[{request_id}][handle_audio_blob] 嘗試在 {temp_dir} 創建臨時 .webm 檔案...")
        try:
            # 使用 delete=False 確保文件在 with 語句結束後不被刪除，以便 process_audio_file 訪問
            # 指定 dir=temp_dir
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False, dir=temp_dir) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name # 獲取完整路徑
            logger.info(f"[{request_id}][handle_audio_blob] 成功將音訊數據寫入臨時檔案: {tmp_file_path}")
            # 檢查文件是否真的創建了
            if not Path(tmp_file_path).exists():
                 logger.error(f"[{request_id}][handle_audio_blob] 寫入後檢查：臨時檔案 {tmp_file_path} 不存在！")
                 socketio.emit('status', '❌ 伺服器檔案系統錯誤 (Write)')
                 return

        except IOError as io_err:
            logger.error(f"[{request_id}][handle_audio_blob] 寫入臨時檔案時發生 IO 錯誤: {io_err}", exc_info=True)
            socketio.emit('status', '❌ 伺服器檔案系統錯誤 (IO)')
            return
        except Exception as tmp_err:
            logger.error(f"[{request_id}][handle_audio_blob] 創建或寫入臨時檔案時發生未知錯誤: {tmp_err}", exc_info=True)
            socketio.emit('status', '❌ 伺服器檔案系統錯誤 (Tmp)')
            return

        # === 步驟 3: 提交異步任務 ===
        logger.info(f"[{request_id}][handle_audio_blob] 準備將 process_audio_file 提交到事件循環...")
        try:
            loop = asyncio.get_event_loop()
            logger.info(f"[{request_id}][handle_audio_blob] 獲取到事件循環: {loop}")
            if loop.is_running():
                logger.info(f"[{request_id}][handle_audio_blob] 事件循環正在運行，提交任務...")
                # 確保傳遞的是有效的 tmp_file_path
                if tmp_file_path and Path(tmp_file_path).exists():
                    future = asyncio.run_coroutine_threadsafe(process_audio_file(tmp_file_path), loop)
                    logger.info(f"[{request_id}][handle_audio_blob] 任務已提交，Future: {future}")
                    # 可以選擇性地添加回調來檢查任務是否成功提交或執行
                    # future.add_done_callback(lambda f: logger.info(f"[{request_id}] Async task completed. Result/Exception: {f.result() if not f.cancelled() else 'Cancelled'}"))
                else:
                    logger.error(f"[{request_id}][handle_audio_blob] 錯誤：臨時檔案路徑無效或檔案不存在，無法提交任務。Path: {tmp_file_path}")
                    socketio.emit('status', '❌ 伺服器內部錯誤 (File Path)')
                    # 需要手動清理已創建但未處理的檔案 (如果 tmp_file_path 有值)
                    if tmp_file_path and Path(tmp_file_path).exists():
                         try: os.remove(tmp_file_path)
                         except OSError as e: logger.warning(f"[{request_id}] 手動清理 {tmp_file_path} 失敗: {e}")
            else:
                logger.warning(f"[{request_id}][handle_audio_blob] 事件循環未運行！無法處理音訊。")
                socketio.emit('status', '❌ 伺服器內部錯誤 (Loop)')
                # 同樣需要手動清理
                if tmp_file_path and Path(tmp_file_path).exists():
                     try: os.remove(tmp_file_path)
                     except OSError as e: logger.warning(f"[{request_id}] 手動清理 {tmp_file_path} 失敗: {e}")

        except Exception as submit_err:
            logger.error(f"[{request_id}][handle_audio_blob] 提交異步任務時發生錯誤: {submit_err}", exc_info=True)
            socketio.emit('status', '❌ 伺服器內部錯誤 (Async Submit)')
            # 清理
            if tmp_file_path and Path(tmp_file_path).exists():
                try: os.remove(tmp_file_path)
                except OSError as e: logger.warning(f"[{request_id}] 清理 {tmp_file_path} 失敗: {e}")


    except Exception as outer_err:
        # 捕獲 handle_audio_blob 函數自身的其他未預期錯誤
        logger.error(f"[{request_id}][handle_audio_blob] 處理 audio_blob 事件時發生頂層錯誤: {outer_err}", exc_info=True)
        socketio.emit('status', '❌ 伺服器發生嚴重錯誤')
        # 清理 (如果 tmp_file_path 已賦值)
        if tmp_file_path and Path(tmp_file_path).exists():
             try: os.remove(tmp_file_path)
             except OSError as e: logger.warning(f"[{request_id}] 清理 {tmp_file_path} 失敗: {e}")

    # 注意：由於 process_audio_file 是異步運行的，handle_audio_blob 函數會在這裡結束，
    # 不會等待 process_audio_file 完成。process_audio_file 內部的 finally 塊負責清理它創建的 .wav 檔案。
    # 而這裡創建的 .webm 檔案，如果成功提交給 process_audio_file，則由 process_audio_file 的 finally 負責清理。
    # 如果提交失敗或在此函數中出錯，則需要在上面的錯誤處理中手動清理 .webm 檔案。


async def process_audio_file(file_path):
    global is_active
    input_path = Path(file_path)
    pcm_path = input_path.with_suffix('.wav') # 更簡潔的寫法
    client = None # 初始化確保 finally 可以檢查

    try:
        # 轉換為 Transcribe 要求的 PCM WAV 格式
        command = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(pcm_path)
        ]
        logger.info(f"[process_audio_file] 執行 FFmpeg: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"[process_audio_file] FFmpeg 轉換失敗 (Code: {process.returncode}): {stderr.decode(errors='ignore')}")
            socketio.emit('status', '❌ 音訊轉檔失敗')
            # 恢復前端狀態
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif' # 用 listening 或 idle
            socketio.emit('expression', expression)
            return # 提前結束
        else:
            logger.info(f"[process_audio_file] FFmpeg 轉換成功: {pcm_path}")

        # 讀取轉換後的 PCM 數據
        with open(pcm_path, 'rb') as f:
            pcm_data = f.read()
        logger.info(f"[process_audio_file] 讀取 WAV 檔案大小: {len(pcm_data)} bytes")
        if len(pcm_data) == 0:
            logger.warning("[process_audio_file] WAV 檔案為空，可能轉換有問題或原始音檔無聲")
            socketio.emit('status', '🟡 未偵測到有效聲音')
            # 恢復前端狀態
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression)
            return # 提前結束


        client = TranscribeStreamingClient(region=REGION)
        # 開始串流轉錄
        stream = await client.start_stream_transcription(
            language_code="zh-TW",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        logger.info("[process_audio_file] 啟動 Transcribe 串流")

        # 將音訊數據分塊發送
        async def write_chunks():
            chunk_size = 8000
            nonlocal pcm_data # 確保能訪問外層的 pcm_data
            try:
                total_sent = 0
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i+chunk_size]
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    total_sent += len(chunk)
                    # logger.debug(f"Sent chunk {i//chunk_size + 1}, size: {len(chunk)}")
                    await asyncio.sleep(0.1) # 短暫延遲
                await stream.input_stream.end_stream()
                logger.info(f"[process_audio_file] 音訊串流發送完畢 (Total: {total_sent} bytes)")
            except Exception as e:
                logger.error(f"[process_audio_file] 發送音訊串流時出錯: {e}", exc_info=True)
                # 嘗試安全關閉 stream
                try: await stream.input_stream.end_stream()
                except: pass
                raise # 向上拋出錯誤

        # 處理轉錄結果
        async def read_results():
            handler = MyTranscriptHandler(stream.output_stream)
            try:
                await handler.handle_events()
                logger.info("[process_audio_file] Transcribe 結果處理完畢")
            except Exception as e:
                logger.error(f"[process_audio_file] 處理 Transcribe 結果時出錯: {e}", exc_info=True)
                raise # 向上拋出錯誤

        # 並發執行寫入和讀取
        # 使用 gather 確保兩者都完成或其中之一出錯時能正確處理
        await asyncio.gather(write_chunks(), read_results())
        logger.info("[process_audio_file] Transcribe 串流處理完成")


    except Exception as e:
        logger.error(f"[process_audio_file] 整體音訊處理失敗：{e}", exc_info=True)
        socketio.emit('status', f'❌ 語音辨識失敗')
        # 出錯後確保前端能回到某個穩定狀態
        current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
        socketio.emit('status', current_status)
        expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
        socketio.emit('expression', expression)
    finally:
        # 清理暫存檔案
        try:
            if input_path.exists():
                os.remove(input_path)
                logger.info(f"[process_audio_file] 已刪除暫存檔：{input_path}")
            if pcm_path.exists():
                os.remove(pcm_path)
                logger.info(f"[process_audio_file] 已刪除 WAV 檔：{pcm_path}")
        except Exception as e:
            logger.warning(f"[process_audio_file] 清理暫存檔失敗: {e}")
        # 確保 Transcribe client 被關閉 (雖然 streaming client 可能沒有顯式的 close)
        # client = None


# --- 新增：根據文字處理意圖 ---
# (沿用上次邏輯，包含狀態處理、任務取消、SocketIO 更新)
async def handle_intent_from_text(text: str):
    global is_active, current_task

    socketio.emit('status', f"💬 收到: \"{text}\"，分析意圖...")
    socketio.emit('expression', '/static/animations/thinking.gif')
    socketio.emit('user_query', text) # 立即顯示用戶輸入

    intent = await classify_intent(text)
    logger.info(f"[handle_intent_from_text] 文字: '{text}', 意圖: {intent}")

    with is_active_lock:
        if intent == "START":
            if not is_active:
                is_active = True
                logger.info("[handle_intent_from_text] 系統啟動")
                with current_task_lock:
                    if current_task and not current_task.done():
                        logger.info("[handle_intent_from_text] (START) 取消舊任務...")
                        current_task.cancel()
                        current_task = None
                socketio.emit('status', '🟢 已啟動，請說指令...')
                socketio.emit('expression', '/static/animations/listening.gif') # 使用 listening
            else:
                logger.info("[handle_intent_from_text] 系統已啟動，忽略 START 指令")
                socketio.emit('status', '🟢 已啟動，請說指令...') # 維持啟動狀態提示
                socketio.emit('expression', '/static/animations/listening.gif')

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
                socketio.emit('expression', '/static/animations/idle.gif')
            else:
                logger.info("[handle_intent_from_text] 系統已關閉，忽略 STOP 指令")
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...') # 維持待機狀態提示
                socketio.emit('expression', '/static/animations/idle.gif')

        elif intent == "INTERRUPT":
            if is_active:
                logger.info("[handle_intent_from_text] 收到打斷指令")
                interrupted = False
                with current_task_lock:
                    if current_task and not current_task.done():
                        logger.info("[handle_intent_from_text] (INTERRUPT) 取消進行中任務...")
                        current_task.cancel()
                        current_task = None # 清除引用
                        interrupted = True

                if interrupted:
                    socketio.emit('status', '🟡 已中斷，請說新指令...')
                    socketio.emit('expression', '/static/animations/listening.gif') # 等待新指令
                else:
                    logger.info("[handle_intent_from_text] (INTERRUPT) 無任務可中斷，等待新指令...")
                    socketio.emit('status', '🟢 已啟動，請說指令...') # 回到等待指令狀態
                    socketio.emit('expression', '/static/animations/listening.gif')
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
                # ⭐ 即使忽略指令，也要確保前端能重新監聽
                #    前端會在 status 更新後，根據狀態決定是否 startListening

        elif intent == "IGNORE":
            logger.info("[handle_intent_from_text] 忽略無法分類或無效的指令")
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status) # 回復當前狀態提示
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression) # 回復對應表情

    # ⭐ 不在此處觸發前端重新監聽，讓前端的事件回調（如 audio_url onended）負責


# --- 任務處理 ---
# (沿用上次的修改，使用 asyncio.to_thread，修正 URL 生成方式)
async def handle_text(text: str):
    global is_active
    output_dir = Path("./history_result")
    output_dir.mkdir(exist_ok=True)
    audio_path = None
    generated_text = None

    try:
        logger.info(f"[handle_text] 開始處理命令：{text}")

        # --- TaskClassifier (如果需要) ---
        task_classifier = TaskClassifier()
        classify_func = retry_sync(retries=3, delay=1)(task_classifier.classify_task)
        task_type, _ = await asyncio.to_thread(classify_func, text)
        logger.info(f"[handle_text] 任務分類結果：{task_type}")
        # ---------------------------------

        ts = time.strftime('%Y%m%d_%H%M%S')

        # --- 執行任務 ---
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
            if generated_text:
                 logger.info(f"[handle_text] 行動分解結果: {generated_text}")
                 # 選擇性 TTS
                 # audio_path = output_dir / f"output_action_{ts}.mp3"
                 # tts_func = retry_sync(retries=3, delay=1)(PollyTTS().synthesize)
                 # await asyncio.to_thread(tts_func, f"好的，收到行動指令。", str(audio_path))

        else:
             logger.warning(f"[handle_text] 未知的任務類型: {task_type}")
             generated_text = "抱歉，我不太理解這個指令。"
             audio_path = output_dir / f"output_unknown_{ts}.mp3"
             tts_func = retry_sync(retries=3, delay=1)(PollyTTS().synthesize)
             await asyncio.to_thread(tts_func, generated_text, str(audio_path))

        # --- 處理結果 ---
        if generated_text:
            socketio.emit('text_response', generated_text) # 發送文字回覆

        if audio_path and audio_path.exists():
            logger.info(f"[handle_text] 音檔生成完成：{audio_path}")
            # ⭐ 生成相對 URL 給前端
            relative_audio_url = f"/history_result/{audio_path.name}"
            logger.info(f"[handle_text] 生成相對 Audio URL: {relative_audio_url}")
            socketio.emit('expression', '/static/animations/speaking.gif')
            socketio.emit('audio_url', relative_audio_url) # 發送相對 URL
        else:
            # 如果沒有音檔，任務處理完畢後，也需要讓前端知道可以恢復監聽
            logger.info("[handle_text] 無音檔生成，任務處理完畢")
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression)
            # ⭐ 前端會在收到 status 和 expression 後，根據狀態決定是否 startListening

        # 狀態文本由 audio_url 播放完畢或無音檔時直接發送，這裡不發 '✅ 已完成'

    except asyncio.CancelledError:
        logger.info("[handle_text] 任務被取消")
        # 狀態已在 handle_intent_from_text (INTERRUPT) 或 cancellable_socket_handle_text 中處理
        # socketio.emit('status', '🟡 任務已中斷') # 可能重複發送
        # socketio.emit('expression', '/static/animations/listening.gif')
        raise # 重新拋出
    except Exception as e:
        logger.error(f"[handle_text] 處理命令時發生錯誤：{e}", exc_info=True)
        socketio.emit('status', f'❌ 執行命令時出錯')
        # 出錯後恢復狀態
        current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
        socketio.emit('status', current_status)
        expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
        socketio.emit('expression', expression)


# --- 可取消的任務處理入口 ---
# (沿用上次邏輯)
async def cancellable_socket_handle_text(text: str):
    global current_task
    cancelled_previous = False
    with current_task_lock:
        if current_task and not current_task.done():
            logger.info("[cancellable_socket_handle_text] 收到新命令，取消上一個任務...")
            current_task.cancel()
            cancelled_previous = True
            # 短暫等待確保取消生效 (可選)
            # await asyncio.sleep(0.1)

    # 如果取消了上一個任務，給一點時間讓取消的副作用（如狀態更新）發生
    # if cancelled_previous:
    #     await asyncio.sleep(0.2)

    loop = asyncio.get_running_loop()
    logger.info(f"[cancellable_socket_handle_text] 創建新任務來處理命令: {text}")
    # 創建任務並儲存
    task = loop.create_task(handle_text(text), name=f"HandleText_{text[:20]}")
    with current_task_lock:
        current_task = task

    # 可以添加完成回調來清理 current_task 引用 (可選，如果任務正常結束)
    # def _task_done_callback(fut):
    #     global current_task
    #     with current_task_lock:
    #         if current_task == fut: # 確保是同一個任務
    #             current_task = None
    #     try:
    #         fut.result() # 檢查是否有異常
    #         logger.info(f"任務 {fut.get_name()} 正常完成")
    #     except asyncio.CancelledError:
    #         logger.info(f"任務 {fut.get_name()} 被取消")
    #     except Exception as e:
    #         logger.error(f"任務 {fut.get_name()} 執行出錯: {e}", exc_info=True)
    #
    # task.add_done_callback(_task_done_callback)


# --- 路由 ---
@app.route('/')
def index():
    return render_template_string(HTML)

# (沿用上次的安全路由)
@app.route('/history_result/<path:filename>')
def get_audio(filename):
    logger.debug(f"請求音檔: {filename}")
    directory = Path('history_result').resolve()
    try:
        # 安全地檢查路徑
        safe_path = (directory / filename).resolve()
        if safe_path.is_file() and safe_path.parent == directory:
             logger.debug(f"提供檔案: {safe_path}")
             return send_from_directory(directory, filename) # Flask 會處理 Content-Type
        else:
             logger.warning(f"拒絕存取不在允許目錄的檔案: {filename}")
             return "Forbidden", 403
    except FileNotFoundError:
        logger.error(f"請求的音檔不存在: {filename}")
        return "Not Found", 404
    except Exception as e:
        logger.error(f"提供音檔 '{filename}' 時發生錯誤: {e}", exc_info=True)
        return "Internal Server Error", 500

# --- 主程式 ---
if __name__ == '__main__':
    os.makedirs('history_result', exist_ok=True)
    logger.info("啟動 SocketIO Server on 0.0.0.0:5000")
    logger.info(f"Flask Server Name Config: {app.config.get('SERVER_NAME')}")
    # 使用 eventlet 或 gevent 通常比 Werkzeug 的開發伺服器更適合 SocketIO
    # 例如: pip install eventlet
    # socketio.run(app, host='0.0.0.0', port=5000) # 使用 eventlet (如果已安裝)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True) # 使用 Werkzeug 開發伺服器
    # 注意：allow_unsafe_werkzeug=True 不應用於生產環境！
    # debug=True, use_reloader=False 適合開發調試
