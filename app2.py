import os
import threading
import asyncio
import time
import logging
import base64
import tempfile
import subprocess
import json # <-- 新增 json 導入
from pathlib import Path # <-- 新增 pathlib 導入
# from urllib.parse import urljoin # <-- 如果不使用 external=True 的 url_for 可以不用
# from flask import request # <-- 如果沒有其他地方用到，可以移除
from flask import Flask, render_template_string, send_from_directory, url_for
from flask_socketio import SocketIO
from tools.retry_utils import retry_sync
# 請確保 rag_chat, tts, agent, task_classification 模組和其中的類別存在且可導入
from rag_chat.rag import RAGPipeline, WebSearcher, ConversationalModel
from rag_chat.chat import Chatbot
from tts.tts import PollyTTS
from agent.action_decompose import ActionDecomposer
from task_classification.task_classification import TaskClassifier
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import boto3 # <-- 新增 boto3 導入
from botocore.config import Config # <-- 新增 Config 導入

# --- 環境初始化 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
# app.config['SERVER_NAME'] = 'localhost:5000' # 如果不需要 url_for(_external=True) 可以不設定或謹慎設定
app.config['PREFERRED_URL_SCHEME'] = 'https' # 告知 Flask 使用 https 生成外部 URL (如果需要的話)

socketio = SocketIO(app, cors_allowed_origins="*")

current_task = None # 用於追蹤和取消當前正在處理 handle_text 的任務
current_task_lock = threading.Lock() # 保護 current_task 變數

is_active = False # <-- 新增：系統啟動狀態，只有 active 時才響應 COMMAND
is_active_lock = threading.Lock() # <-- 新增：保護 is_active 變數

# ---------- Bedrock 參數 ----------
# 確保 AWS 憑證已配置 (環境變數、IAM Role、~/.aws/credentials 等)
REGION = "us-west-2" # <--- 確認這是你的 AWS 區域
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # <--- 確認模型 ID 和權限
BEDROCK_CONFIG = Config(
    region_name=REGION,
    connect_timeout=10, # 連線超時 (秒)
    read_timeout=300    # 讀取超時 (秒)
)
# ⭐ 請確保 BEDROCK 實例在使用時才初始化，或確認 boto3 在多線程環境中的行為
# BEDROCK   = boto3.client("bedrock-runtime", config=BEDROCK_CONFIG)
# 將 BEDROCK 客戶端初始化移到需要的地方，或者確認其線程安全性。
# 在 classify_intent 中調用 boto3.client 是可以的，它通常是線程安全的。

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

# ---------- Bedrock 分類 함수 ----------
# 需要依賴 boto3 模組
async def classify_intent(text: str) -> str:
    """使用 Bedrock 模型分類文字意圖。"""
    # 在函數內部獲取客戶端以避免潛在的線程問題
    bedrock_runtime = boto3.client("bedrock-runtime", config=BEDROCK_CONFIG)

    user_prompt = _CLASSIFY_PROMPT.format(text=text.replace('"', '\\"'))
    logger.info(f"[classify_intent] 準備分類文字：{text[:50]}...") # 避免日誌過長

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10, # 分類只需要很少的 tokens
        "temperature": 0, # 意圖分類需要確定的結果，溫度設為 0
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    # 為了在 asyncio 中執行同步的 invoke_model，我們定義一個內部同步函數並使用 asyncio.to_thread
    @retry_sync(retries=2, delay=0.5) # 添加重試機制
    def _invoke_sync():
        try:
            resp = bedrock_runtime.invoke_model(
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
                    return "COMMAND" # 非預期結果也歸類為 COMMAND
            else:
                logger.warning(f"[classify_intent] Bedrock 回應格式錯誤或無內容，歸類為 COMMAND")
                return "COMMAND" # 回應格式錯誤也歸類為 COMMAND
        except Exception as e:
            logger.error(f"[classify_intent] Bedrock invoke 失敗：{e}", exc_info=True)
            raise # 重新拋出異常以觸發 retry_sync

    try:
        # 使用 asyncio.to_thread 在一個單獨的線程中執行同步的 _invoke_sync
        intent = await asyncio.to_thread(_invoke_sync)
        return intent
    except Exception as e:
        # 如果 retry 後仍然失敗
        logger.error(f"[classify_intent] Bedrock 分類重試後仍然失敗：{e}", exc_info=True)
        return "IGNORE" # 分類失敗則忽略該語句

# -----------------------------------


# --- 啟動時檢查 ffmpeg ---
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    logger.info("✅ ffmpeg 檢查成功")
except Exception:
    logger.error("❌ 找不到 ffmpeg，請安裝 ffmpeg。")
    raise # 如果找不到 ffmpeg，終止程式

# --- Transcript Handler ---
class MyTranscriptHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                text = result.alternatives[0].transcript.strip()
                if text:
                    logger.info(f"[TranscribeHandler] 轉出文字：{text}")
                    # ⭐ 修改：轉錄出文字後，呼叫意圖處理函數
                    await handle_intent_from_text(text) # handle_intent_from_text 會決定下一步

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
    """安全地刪除 history_result 目錄下的檔案。"""
    try:
        # 獲取 history_result 目錄的絕對路徑
        base_dir = Path('history_result').resolve()
        # 組合用戶提供的文件名，並解析為絕對路徑
        path_to_delete = (base_dir / filename).resolve()

        # 檢查：檔案是否存在 且 檔案的父目錄是 history_result 目錄
        # 這防止了通過 '..', 符號鏈接等方式刪除其他目錄下的檔案
        if path_to_delete.is_file() and path_to_delete.parent == base_dir:
            os.remove(path_to_delete)
            logger.info(f"[delete_audio] 已安全刪除檔案：{path_to_delete}")
        else:
            logger.warning(f"[delete_audio] 嘗試刪除無效或不安全的檔案路徑：{filename}")
    except Exception as e:
        logger.error(f"[delete_audio] 刪除檔案 '{filename}' 失敗：{e}", exc_info=True)


# --- 音訊處理 ---

@socketio.on('audio_blob')
def handle_audio_blob(base64_audio):
    """接收前端傳來的音訊 Base64 數據，解碼並提交給異步處理。"""
    # 增加日誌記錄，標識每次調用
    request_id = f"req_{time.monotonic_ns()}" # 創建一個簡單的請求 ID
    logger.info(f"[{request_id}][handle_audio_blob] 收到 audio_blob 事件")

    # 檢查收到的數據類型和初步內容
    if not isinstance(base64_audio, str):
        logger.error(f"[{request_id}][handle_audio_blob] 錯誤：收到的 base64_audio 不是字串，類型為 {type(base64_audio)}")
        socketio.emit('status', '❌ 錯誤：音訊數據格式不對')
        # 恢復前端狀態
        with is_active_lock:
             current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
             socketio.emit('status', current_status)
             expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
             socketio.emit('expression', expression)
        return
    #logger.info(f"[{request_id}][handle_audio_blob] 收到 Base64 字串，前 50 字元: {base64_audio[:50]}...")
    logger.info(f"[{request_id}][handle_audio_blob] Base64 字串總長度: {len(base64_audio)}")

    # ⭐ 收到音訊後馬上切換成 thinking.gif (這是處理中的通用狀態)
    socketio.emit('expression', '/static/animations/thinking.gif')

    tmp_file_path = None # 初始化確保 finally 可以檢查

    try:
        # === 步驟 1: Base64 解碼 ===
        logger.info(f"[{request_id}][handle_audio_blob] 嘗試 Base64 解碼...")
        try:
            audio_data = base64.b64decode(base64_audio)
            logger.info(f"[{request_id}][handle_audio_blob] Base64 解碼成功，得到 {len(audio_data)} bytes 的音訊數據")
        except Exception as decode_e: # 捕獲更廣泛的異常
            logger.error(f"[{request_id}][handle_audio_blob] Base64 解碼失敗: {decode_e}", exc_info=True)
            socketio.emit('status', '❌ 無效的音訊數據')
            # 恢復前端狀態
            with is_active_lock:
                 current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                 socketio.emit('status', current_status)
                 expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                 socketio.emit('expression', expression)
            return # 解碼失敗，無法繼續

        # === 步驟 2: 創建並寫入臨時檔案 (.webm) ===
        temp_dir = Path("./temp_audio") # 建議使用一個專用臨時目錄
        try:
            # 使用 exist_ok=True 避免目錄已存在時報錯
            temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{request_id}][handle_audio_blob] 確保臨時目錄存在: {temp_dir.resolve()}")
        except OSError as dir_err:
             logger.error(f"[{request_id}][handle_audio_blob] 無法創建或訪問臨時目錄 {temp_dir.resolve()}: {dir_err}", exc_info=True)
             socketio.emit('status', '❌ 伺服器檔案系統錯誤 (Dir)')
             # 恢復前端狀態
             with is_active_lock:
                  current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                  socketio.emit('status', current_status)
                  expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                  socketio.emit('expression', expression)
             return

        logger.info(f"[{request_id}][handle_audio_blob] 嘗試在 {temp_dir} 創建臨時 .webm 檔案...")
        try:
            # 使用 delete=False 確保文件在 with 語句結束後不被刪除，以便 process_audio_file 訪問
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False, dir=temp_dir) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name # 獲取完整路徑
            logger.info(f"[{request_id}][handle_audio_blob] 成功將音訊數據寫入臨時檔案: {tmp_file_path}")
            # 檢查文件是否真的創建了 (額外檢查)
            if not Path(tmp_file_path).exists():
                 logger.error(f"[{request_id}][handle_audio_blob] 寫入後檢查：臨時檔案 {tmp_file_path} 不存在！")
                 socketio.emit('status', '❌ 伺服器檔案系統錯誤 (Write)')
                  # 恢復前端狀態
                 with is_active_lock:
                      current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                      socketio.emit('status', current_status)
                      expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                      socketio.emit('expression', expression)
                 return

        except Exception as file_err: # 捕獲創建或寫入檔案時的任何異常
            logger.error(f"[{request_id}][handle_audio_blob] 創建或寫入臨時檔案時發生錯誤: {file_err}", exc_info=True)
            socketio.emit('status', '❌ 伺服器檔案系統錯誤')
             # 恢復前端狀態
            with is_active_lock:
                 current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                 socketio.emit('status', current_status)
                 expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                 socketio.emit('expression', expression)
            return


        # === 步驟 3: 提交異步任務 ===
        logger.info(f"[{request_id}][handle_audio_blob] 準備將 process_audio_file 提交到事件循環...")
        try:
            # ⭐ 嘗試獲取當前線程的 asyncio loop
            #    這部分需要你的主應用程式正確運行一個 asyncio loop
            loop = asyncio.get_event_loop()
            logger.info(f"[{request_id}][handle_audio_blob] 獲取到事件循環: {loop}")

            if loop.is_running():
                logger.info(f"[{request_id}][handle_audio_blob] 事件循環正在運行，提交任務...")
                # 確保 tmp_file_path 有效並存在，雖然前面已檢查過一次
                if tmp_file_path and Path(tmp_file_path).exists():
                    # ⭐ 使用 run_coroutine_threadsafe 提交到異步循環
                    #    並傳遞 request_id 和 tmp_file_path
                    future = asyncio.run_coroutine_threadsafe(process_audio_file(tmp_file_path, request_id), loop)
                    logger.info(f"[{request_id}][handle_audio_blob] 任務已提交，Future: {future}")
                    # 任務已成功提交，此 handler 可以結束
                    return
                else:
                    # 這是一個異常情況，文件應當存在
                    logger.error(f"[{request_id}][handle_audio_blob] 錯誤：臨時檔案路徑無效或檔案提交前丟失。Path: {tmp_file_path}")
                    socketio.emit('status', '❌ 伺服器內部錯誤 (File Submit)')
                    # 恢復前端狀態
                    with is_active_lock:
                         current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                         socketio.emit('status', current_status)
                         expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                         socketio.emit('expression', expression)
                    # 手動清理 (如果 tmp_file_path 已賦值且存在)
                    if tmp_file_path and Path(tmp_file_path).exists():
                         try: os.remove(tmp_file_path)
                         except OSError as e: logger.warning(f"[{request_id}] 手動清理 {tmp_file_path} 失敗: {e}")
                    return # 返回

            else:
                # Loop 未運行
                logger.warning(f"[{request_id}][handle_audio_blob] 事件循環未運行！無法處理音訊。")
                socketio.emit('status', '❌ 伺服器內部錯誤 (Loop Not Running)')
                # 恢復前端狀態
                with is_active_lock:
                     current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                     socketio.emit('status', current_status)
                     expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                     socketio.emit('expression', expression)
                # 手動清理因為任務未提交
                if tmp_file_path and Path(tmp_file_path).exists():
                     try: os.remove(tmp_file_path)
                     except OSError as e: logger.warning(f"[{request_id}] 手動清理 {tmp_file_path} 失敗 (Loop Not Running Cleanup): {e}")
                return # 返回

        except Exception as submit_err: # 捕獲在獲取 loop 或提交任務時可能發生的任何錯誤
            logger.error(f"[{request_id}][handle_audio_blob] 獲取循環或提交異步任務時發生錯誤: {submit_err}", exc_info=True)
            socketio.emit('status', '❌ 伺服器內部錯誤 (Async Submit Error)')
            # 恢復前端狀態
            with is_active_lock:
                 current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                 socketio.emit('status', current_status)
                 expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                 socketio.emit('expression', expression)
            # 手動清理因為任務未提交
            if tmp_file_path and Path(tmp_file_path).exists():
                try: os.remove(tmp_file_path)
                except OSError as e: logger.warning(f"[{request_id}] 清理 {tmp_file_path} 失敗 (Submit Error Cleanup): {e}")
            return # 返回

    # 捕獲前面未捕獲的任何其他頂層錯誤 (不太可能，因為前面的 except 已經很廣泛)
    except Exception as outer_err:
        logger.error(f"[{request_id}][handle_audio_blob] 處理 audio_blob 事件時發生未預期錯誤: {outer_err}", exc_info=True)
        socketio.emit('status', '❌ 伺服器發生未預期嚴重錯誤')
        # 恢復前端狀態
        with is_active_lock:
             current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
             socketio.emit('status', current_status)
             expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
             socketio.emit('expression', expression)
        # 此處通常不需要清理 tmp_file_path，因為錯誤很可能發生在文件創建之前

    logger.info(f"[{request_id}][handle_audio_blob] 函數結束 (任務已提交或處理失敗)")


# ⭐ 修改 process_audio_file，接收 input_path_str 和 request_id，並在 finally 中清理 .webm 和 .wav
#    並在 Transcribe 收到完整文字後呼叫 handle_intent_from_text
async def process_audio_file(input_path_str: str, request_id: str):
    """
    處理音訊檔案 (轉檔, Transcribe)，並在轉錄完成後觸發意圖處理。
    負責清理臨時檔案 (.webm 和 .wav)。
    運行在 asyncio event loop 中。
    """
    global is_active # 訪問 is_active 來恢復前端狀態 (在錯誤情況下)
    logger.info(f"[{request_id}][process_audio_file] 開始處理檔案: {input_path_str}")
    input_path = Path(input_path_str) # .webm 檔案
    pcm_path = input_path.with_suffix('.wav') # 將要生成的 .wav 檔案路徑
    client = None # 初始化 Transcribe 客戶端

    try:
        # 確保輸入文件存在 (在 handle_audio_blob 中已檢查過一次，這裡作為異步任務的額外檢查)
        if not input_path.exists():
            logger.error(f"[{request_id}][process_audio_file] 輸入檔案不存在或已丟失: {input_path}")
            socketio.emit('status', '❌ 輸入音訊檔案丟失')
             # 恢復前端狀態
            with is_active_lock:
                current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                socketio.emit('status', current_status)
                expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                socketio.emit('expression', expression)
            return # 提前結束，finally 會清理


        # 轉換為 Transcribe 要求的 PCM WAV 格式
        command = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(pcm_path)
        ]
        logger.info(f"[{request_id}][process_audio_file] 執行 FFmpeg: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"[{request_id}][process_audio_file] FFmpeg 轉換失敗 (Code: {process.returncode}): {stderr.decode(errors='ignore')}")
            socketio.emit('status', '❌ 音訊轉檔失敗')
            # 恢復前端狀態 (錯誤發生時)
            with is_active_lock:
                current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                socketio.emit('status', current_status)
                expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                socketio.emit('expression', expression)
            return # 轉換失敗，結束處理流程


        # 讀取轉換後的 PCM 數據
        try:
            # 檢查 WAV 檔案是否存在且非空
            if not pcm_path.exists() or pcm_path.stat().st_size == 0:
                 logger.warning(f"[{request_id}][process_audio_file] 轉換後的 WAV 檔案不存在或為空: {pcm_path}")
                 socketio.emit('status', '🟡 未偵測到有效聲音')
                 # 恢復前端狀態 (無聲音時)
                 with is_active_lock:
                      current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                      socketio.emit('status', current_status)
                      expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                      socketio.emit('expression', expression)
                 return # WAV 空檔案，結束處理流程

            with open(pcm_path, 'rb') as f:
                pcm_data = f.read()
            logger.info(f"[{request_id}][process_audio_file] 讀取 WAV 檔案大小: {len(pcm_data)} bytes")


        except FileNotFoundError: # 額外捕獲文件讀取錯誤
             logger.error(f"[{request_id}][process_audio_file] 讀取轉換後的 WAV 檔案失敗 (FileNotFound): {pcm_path}", exc_info=True)
             socketio.emit('status', '❌ 找不到轉換後的音訊檔案')
             # 恢復前端狀態
             with is_active_lock:
                  current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                  socketio.emit('status', current_status)
                  expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                  socketio.emit('expression', expression)
             return # 文件讀取失敗，結束處理流程
        except Exception as read_err:
             logger.error(f"[{request_id}][process_audio_file] 讀取 WAV 檔案失敗: {read_err}", exc_info=True)
             socketio.emit('status', '❌ 讀取音訊檔案失敗')
             # 恢復前端狀態
             with is_active_lock:
                  current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                  socketio.emit('status', current_status)
                  expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                  socketio.emit('expression', expression)
             return # 文件讀取失敗，結束處理流程


        # Transcribe 串流處理
        client = TranscribeStreamingClient(region=REGION)
        stream = await client.start_stream_transcription(
            language_code="zh-TW",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        logger.info(f"[{request_id}][process_audio_file] 啟動 Transcribe 串流")

        # 將音訊數據分塊發送給 Transcribe
        async def write_chunks():
            chunk_size = 8000 # 建議較小的 chunk size 以降低延遲
            nonlocal pcm_data, stream, request_id # 確保能訪問外層的變量
            try:
                total_sent = 0
                # Transcribe 串流通常期望每隔一段時間接收音訊
                # 如果一次性發送大量數據，可能需要調整 Transcribe 參數或發送間隔
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i+chunk_size]
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    total_sent += len(chunk)
                    #logger.debug(f"[{request_id}] Sent chunk {i//chunk_size + 1}, size: {len(chunk)}")
                    # 避免過於頻繁發送，可以根據需要調整或移除 sleep
                    # Transcribe 通常處理得很快，但如果音訊很長，可能需要考慮流量控制
                    await asyncio.sleep(0.005) # 短暫延遲

                await stream.input_stream.end_stream() # 發送結束信號
                logger.info(f"[{request_id}][process_audio_file] 音訊串流發送完畢 (Total: {total_sent} bytes)")
            except Exception as e:
                logger.error(f"[{request_id}][process_audio_file] 發送音訊串流時出錯: {e}", exc_info=True)
                try: await stream.input_stream.end_stream() # 嘗試安全關閉
                except: pass
                raise # 向上拋出錯誤給 gather

        # 處理 Transcribe 的轉錄結果
        async def read_results():
            nonlocal stream, request_id # 確保能訪問 stream 和 request_id
            # MyTranscriptHandler 的 handle_transcript_event 方法會接收到結果
            # 並且我們修改了它，讓它呼叫 handle_intent_from_text
            handler = MyTranscriptHandler(stream.output_stream)
            try:
                await handler.handle_events() # 開始接收並處理事件
                logger.info(f"[{request_id}][process_audio_file] Transcribe 結果處理完畢")
            except Exception as e:
                logger.error(f"[{request_id}][process_audio_file] 處理 Transcribe 結果時出錯: {e}", exc_info=True)
                raise # 向上拋出錯誤給 gather

        # 並發執行音訊發送和結果接收
        await asyncio.gather(write_chunks(), read_results())
        logger.info(f"[{request_id}][process_audio_file] Transcribe 串流處理完成")

        # Transcribe 完成後，handle_intent_from_text 已經根據分類結果處理了狀態或觸發了 handle_command
        # 如果整個流程走到這裡沒有拋出異常，表示 Transcribe 部分成功完成並觸發了後續流程。
        # 前端狀態的最終恢復將取決於 handle_intent_from_text -> handle_command (如果觸發了) 的結果或 TTS 播放結束事件。

    except Exception as process_err: # 捕獲整個 process_audio_file 異步流程中的錯誤
        logger.error(f"[{request_id}][process_audio_file] 音訊處理流程中發生錯誤: {process_err}", exc_info=True)
        socketio.emit('status', '❌ 音訊處理失敗')
        # 在錯誤發生後恢復前端狀態
        with is_active_lock:
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression)

    finally:
        # --- 確保清理臨時檔案 (.webm 和 .wav) ---
        logger.info(f"[{request_id}][process_audio_file] 進入 finally 塊，清理臨時檔案...")
        temp_dir = Path("./temp_audio").resolve() # 再次獲取並解析臨時目錄的安全路徑

        # 清理輸入的 .webm 檔案
        if input_path and input_path.is_file() and input_path.parent == temp_dir:
            try:
                os.remove(input_path)
                logger.info(f"[{request_id}][process_audio_file] 已清理輸入檔案: {input_path}")
            except OSError as e:
                logger.warning(f"[{request_id}][process_audio_file] 清理輸入檔案 {input_path} 失敗: {e}", exc_info=True)
        else:
             # 如果文件不存在，或者不在預期目錄，或者 input_path 為 None (不應該發生但作為防禦)
             if input_path: logger.warning(f"[{request_id}][process_audio_file] 未清理輸入檔案 {input_path}，路徑無效或不在臨時目錄")
             else: logger.warning(f"[{request_id}][process_audio_file] 未清理輸入檔案，input_path 為 None")


        # 清理 FFmpeg 轉換生成的 .wav 檔案
        if pcm_path and pcm_path.is_file() and pcm_path.parent == temp_dir:
             try:
                os.remove(pcm_path)
                logger.info(f"[{request_id}][process_audio_file] 已清理 WAV 檔案: {pcm_path}")
             except OSError as e:
                logger.warning(f"[{request_id}][process_audio_file] 清理 WAV 檔案 {pcm_path} 失敗: {e}", exc_info=True)
        else:
             if pcm_path: logger.warning(f"[{request_id}][process_audio_file] 未清理 WAV 檔案 {pcm_path}，路徑無效或不在臨時目錄")
             else: logger.warning(f"[{request_id}][process_audio_file] 未清理 WAV 檔案，pcm_path 為 None")

        logger.info(f"[{request_id}][process_audio_file] finally 塊結束")


# ⭐ 新增 handle_intent_from_text 函數，負責意圖分類和流程控制 ⭐
async def handle_intent_from_text(text: str):
    """
    接收 Transcribe 轉錄的文字，使用 Bedrock 判斷意圖，
    並根據意圖控制系統狀態 (is_active) 及決定是否觸發後續的 COMMAND 處理流程。
    這個函數運行在 asyncio event loop 中。
    """
    global is_active # 需要訪問和修改全局變量 is_active

    logger.info(f"[handle_intent_from_text] 收到轉錄文字：'{text}'")
    # 避免處理空字符串或只有空白符的字符串
    if not text or not text.strip():
        logger.warning("[handle_intent_from_text] 收到空或空白文字，忽略。")
        # 收到無效文字，恢復到根據 is_active 狀態的預設等待狀態
        with is_active_lock:
             current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
             socketio.emit('status', current_status)
             expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
             socketio.emit('expression', expression)
        return

    # 發送用戶的原始查詢到前端顯示
    socketio.emit('user_query', text)

    # ⭐ 使用 Bedrock 分類意圖 ⭐
    # classify_intent 是 async 函數，直接 await 呼叫
    # 如果 classify_intent 失敗 (例如 Bedrock 服務問題)，會返回 "IGNORE"
    intent = await classify_intent(text)

    logger.info(f"[handle_intent_from_text] 文字：'{text[:50]}...' -> 分類結果：{intent} (系統狀態: {'啟動' if is_active else '待機'})")

    # 發送分類結果狀態到前端 (可選，作為除錯或用戶反饋)
    # socketio.emit('status', f'✨ 意圖：{intent}')

    # --- ⭐ 根據意圖處理邏輯 ⭐ ---
    # 在讀取和修改 is_active 時使用鎖，確保線程安全 (因為 socketio handlers 運行在不同線程)
    with is_active_lock:

        if intent == "START":
            if not is_active:
                is_active = True
                logger.info("[handle_intent_from_text] 偵測到啟動詞，系統啟動。")
                socketio.emit('status', '🟢 已啟動，請說指令...')
                socketio.emit('expression', '/static/animations/listening.gif')
            else:
                logger.info("[handle_intent_from_text] 偵測到啟動詞，但系統已在啟動狀態。")
                # 即使已啟動，也恢復到聆聽狀態的動畫和文字
                socketio.emit('status', '🟢 已啟動，請說指令...')
                socketio.emit('expression', '/static/animations/listening.gif')
            # START 意圖只改變狀態，不觸發 handle_text 流程

        elif intent == "STOP":
            if is_active:
                is_active = False
                logger.info("[handle_intent_from_text] 偵測到結束詞，系統關閉。")
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...')
                socketio.emit('expression', '/static/animations/idle.gif')
                # TODO: 如果有正在進行的任務 (如TTS播放或 handle_text 流程)，可能需要一個機制來中斷它
                # 這可能涉及取消 current_task
                with current_task_lock:
                     if current_task and not current_task.done():
                          logger.info("[handle_intent_from_text] 收到 STOP 意圖，嘗試取消當前 COMMAND 任務...")
                          current_task.cancel()
                          socketio.emit('status', '🟡 正在取消任務...') # 顯示取消狀態
                # 取消後，恢復為待機狀態已在上面發送
            else:
                 logger.info("[handle_intent_from_text] 偵測到結束詞，但系統已在待機狀態。")
                 # 即使已待機，也恢復到待機狀態的動畫和文字
                 socketio.emit('status', '⏳ 系統待機中，請說啟動詞...')
                 socketio.emit('expression', '/static/animations/idle.gif')
            # STOP 意圖只改變狀態，不觸發 handle_text 流程

        elif intent == "INTERRUPT":
            logger.warning("[handle_intent_from_text] 偵測到中斷詞。功能待實作，嘗試取消當前任務。")
            # 收到中斷，嘗試取消當前正在進行的 COMMAND 任務
            with current_task_lock:
                 if current_task and not current_task.done():
                      logger.info("[handle_intent_from_text] 收到 INTERRUPT 意圖，嘗試取消當前 COMMAND 任務...")
                      current_task.cancel()
                      socketio.emit('status', '🟡 正在取消任務...') # 顯示取消狀態
            # 取消後，恢復到根據 is_active 狀態的預設等待狀態
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression)
            # TODO: 可能需要更精細的中斷邏輯

        # ⭐ 關鍵：如果是 COMMAND 意圖，並且系統處於啟動狀態，才觸發 handle_text 流程 ⭐
        elif intent == "COMMAND":
            if is_active:
                logger.info("[handle_intent_from_text] 系統已啟動並偵測到指令，觸發後續處理...")
                # 發送狀態和動畫，表示正在處理指令 (思考中)
                socketio.emit('status', '🧠 思考中...')
                socketio.emit('expression', '/static/animations/thinking.gif')
                # ⭐ 呼叫 cancellable_socket_handle_text 來執行後續的 TaskClassifier 和 RAG 流程 ⭐
                # cancellable_socket_handle_text 會管理任務取消和在異步循環中運行 handle_text
                # 因為 handle_intent_from_text 已經運行在異步循環中，可以直接 await 這個呼叫
                try:
                    await cancellable_socket_handle_text(text)
                    # handle_text 流程完成後，其內部應該會更新最終狀態和表情
                    # 或者如果 handle_text 沒有發送最終狀態，可以在這裡補一個通用的完成狀態
                    # logger.info(f"[handle_intent_from_text] COMMAND 流程已觸發並完成。")

                except asyncio.CancelledError:
                     logger.info(f"[handle_intent_from_text] COMMAND 任務被取消：'{text}'")
                     socketio.emit('status', '🟡 任務已取消。')
                     # 任務取消後，恢復到根據 is_active 狀態的預設等待狀態
                     current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                     socketio.emit('status', current_status)
                     expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                     socketio.emit('expression', expression)
                except Exception as command_process_err:
                     logger.error(f"[handle_intent_from_text] COMMAND 處理流程失敗 (呼叫 cancellable_socket_handle_text 處): {command_process_err}", exc_info=True)
                     socketio.emit('status', '❌ 指令處理流程錯誤')
                      # 錯誤後恢復狀態
                     current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                     socketio.emit('status', current_status)
                     expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                     socketio.emit('expression', expression)


            else:
                # 系統未啟動時收到 COMMAND 意圖，忽略
                logger.info("[handle_intent_from_text] 偵測到指令，但系統未啟動，忽略。")
                # 恢復到待機狀態
                socketio.emit('status', '⏳ 系統待機中，請說啟動詞...')
                socketio.emit('expression', '/static/animations/idle.gif')


        elif intent == "IGNORE":
             logger.info("[handle_intent_from_text] 分類器忽略了此文字。")
              # 忽略時，恢復到根據 is_active 狀態的預設等待狀態
             current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
             socketio.emit('status', current_status)
             expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
             socketio.emit('expression', expression)

        else:
             # 處理 classify_intent 返回了非預期結果的情況 (理論上不應該發生)
             logger.warning(f"[handle_intent_from_text] 收到未知的意圖分類結果：{intent}。")
             # 恢復到根據 is_active 狀態的預設等待狀態
             current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
             socketio.emit('status', current_status)
             expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
             socketio.emit('expression', expression)


# ⭐ 保持 cancellable_socket_handle_text 函數，它負責管理 handle_text 的任務狀態 ⭐
# 這個函數運行在 asyncio event loop 中，由 handle_intent_from_text 呼叫
async def cancellable_socket_handle_text(text: str):
    """
    取消前一個正在進行的 handle_text 任務，並創建一個新的任務來處理當前文字。
    """
    global current_task # 訪問全局變數
    with current_task_lock: # 使用鎖保護對 current_task 的訪問

        if current_task and not current_task.done():
            logger.info("[cancellable_socket_handle_text] 偵測到新文字，取消上一個任務...")
            current_task.cancel()
            # 等待任務真正取消完成，或者給它一個超時時間
            try:
                await current_task
            except asyncio.CancelledError:
                logger.info("[cancellable_socket_handle_text] 上一個任務已成功取消。")
            except Exception as e:
                 logger.error(f"[cancellable_socket_handle_text] 等待上一個任務取消完成時發生錯誤: {e}", exc_info=True)


        # 獲取當前正在運行的 asyncio loop
        loop = asyncio.get_running_loop()
        logger.info(f"[cancellable_socket_handle_text] 在 loop {loop} 中創建新的 handle_text 任務...")
        # ⭐ 創建並啟動新的 handle_text 任務
        current_task = loop.create_task(handle_text(text))


# ⭐ 保持 handle_text 函數，它負責具體的任務分類和後續 RAG/TTS 流程 ⭐
# 這個函數由 cancellable_socket_handle_text 呼叫，運行在 asyncio event loop 中。
# 它需要處理可能被取消的情況 (asyncio.CancelledError)。
async def handle_text(text: str):
    """
    執行文本的任務分類，並根據分類結果調用相應的 RAG/Chatbot/Action/TTS 流程。
    """
    global is_active # 可能需要在錯誤或完成時根據 is_active 恢復狀態
    logger.info(f"[handle_text] 開始處理文字：{text}")
    # 發送狀態到前端
    # socketio.emit('status', f"📝 偵測到文字：{text}") # 這個狀態由 handle_intent_from_text 發送更合適
    # socketio.emit('user_query', text) # 這個狀態由 handle_intent_from_text 發送

    try:
        # ⭐ Task Classification (這部分可能需要 asyncio.to_thread 包裝，如果它是同步阻塞的)
        task_classifier = TaskClassifier()
        # 使用 asyncio.to_thread 運行同步的 classify_task 方法
        task_type, _ = await asyncio.to_thread(retry_sync(retries=3, delay=1)(task_classifier.classify_task), text)
        logger.info(f"[handle_text] 任務分類結果：{task_type}")

        socketio.emit('expression', '/static/animations/thinking.gif') # 思考中動畫

        audio_path = None
        generated_text = None
        ts = time.strftime('%Y%m%d_%H%M%S')
        history_dir = Path("./history_result") # 使用 Path 對象
        history_dir.mkdir(exist_ok=True) # 確保目錄存在

        # ⭐ 根據任務類型執行相應邏輯 ⭐
        if task_type == "聊天":
            # Chatbot (這部分也可能需要 asyncio.to_thread)
            chat_model = Chatbot(model_id=MODEL_ID) # 確保模型ID正確
            generated_text = await asyncio.to_thread(retry_sync(retries=3, delay=1)(chat_model.chat), text)

            # TTS 合成 (PollyTTS 也可能需要 asyncio.to_thread)
            audio_filename = f"output_chat_{ts}.mp3"
            audio_path = history_dir / audio_filename
            await asyncio.to_thread(retry_sync(retries=3, delay=1)(PollyTTS().synthesize), generated_text, str(audio_path), voice_id='Zhiyu') # 指定語音ID，並將 Path 轉為 string


        elif task_type == "查詢":
            # RAG Pipeline (WebSearcher 和 ConversationalModel)
            # 如果這些內部方法是同步阻塞的，它們的調用也需要包裝在 asyncio.to_thread 中
            web_searcher = WebSearcher(max_results=3, search_depth="advanced", use_top_only=True)
            conversational_model = ConversationalModel(model_id=MODEL_ID) # 確保模型ID正確
            pipeline = RAGPipeline(web_searcher=web_searcher, model=conversational_model)
            generated_text = await asyncio.to_thread(retry_sync(retries=3, delay=1)(pipeline.answer), text)

            # TTS 合成
            audio_filename = f"output_search_{ts}.mp3"
            audio_path = history_dir / audio_filename
            await asyncio.to_thread(retry_sync(retries=3, delay=1)(PollyTTS().synthesize), generated_text, str(audio_path), voice_id='Zhiyu')


        elif task_type == "行動":
            # Action Decomposer
            # 這部分也可能需要 asyncio.to_thread
            action_decomposer = ActionDecomposer()
            generated_text = await asyncio.to_thread(retry_sync(retries=3, delay=1)(action_decomposer.decompose), text)
            # 行動通常沒有直接語音回覆，可能只是文本說明或觸發其他系統動作


        elif task_type == "未知": # 假設 TaskClassifier 可能返回未知類型
             logger.warning(f"[handle_text] 未知任務類型：{task_type}")
             generated_text = "抱歉，我不明白您的意思，請再說一次。"
             # 可以選擇是否合成語音
             audio_filename = f"output_unknown_{ts}.mp3"
             audio_path = history_dir / audio_filename
             await asyncio.to_thread(retry_sync(retries=3, delay=1)(PollyTTS().synthesize), generated_text, str(audio_path), voice_id='Zhiyu')


        else:
             # 處理 TaskClassifier 返回的其他非預期結果
             logger.warning(f"[handle_text] TaskClassifier 返回非預期結果：{task_type}")
             generated_text = "發生內部錯誤，請稍後再試。"
             # 也可以選擇是否合成語音，或者只發送文本


        # 發送生成的文字回覆到前端
        if generated_text:
            socketio.emit('text_response', generated_text)
            logger.info(f"[handle_text] 發送文字回覆：{generated_text[:100]}...")


        # 如果生成了語音檔案且檔案存在
        if audio_path and audio_path.exists(): # 使用 Path 對象檢查是否存在
            logger.info(f"[handle_text] 音檔生成完成：{audio_path}")
            # ⭐ 在這裡獲取 Flask 應用上下文，以便使用 url_for
            with app.app_context():
                # 使用 url_for 生成前端可以訪問的 URL，指向 get_audio 路由
                # _external=False 生成相對路徑，更安全且通常適用於同源請求
                audio_url = url_for('get_audio', filename=audio_path.name, _external=False)
            socketio.emit('expression', '/static/animations/speaking.gif') # 說話動畫
            socketio.emit('audio_url', audio_url)
            logger.info(f"[handle_text] 發送音訊 URL：{audio_url}")

        # 處理完成後的最終狀態更新
        # 如果有發送 audio_url，前端的 player.onended 事件會觸發 startListening 恢復狀態
        # 如果只有文本回覆 (例如 行動 類型)，需要在 handle_text 結束後恢復狀態
        # 或者可以在 handle_intent_from_text 收到 COMMAND 任務完成或取消信號後統一恢復
        # 這裡先補一個通用的完成狀態，實際情況可能需要根據流程微調
        if not audio_path or not audio_path.exists(): # 如果沒有語音回覆
             with is_active_lock: # 根據當前 is_active 狀態恢復前端顯示
                  current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
                  socketio.emit('status', current_status)
                  expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
                  socketio.emit('expression', expression)
             logger.info("[handle_text] COMMAND 處理完成，無語音回覆，已恢復狀態。")
        else:
             # 有語音回覆，狀態恢復交給前端的 audio_url onended 事件
             logger.info("[handle_text] COMMAND 處理完成，等待語音播放結束恢復狀態。")


    except asyncio.CancelledError:
        # 捕獲任務取消異常
        logger.info("[handle_text] 任務被取消")
        # 取消後的前端狀態恢復由 handle_intent_from_text 或 cancellable_socket_handle_text 處理
        raise # 重新拋出異常，讓外層知道任務被取消了
    except Exception as e:
        # 捕獲 handle_text 內部的其他所有異常
        logger.error(f"[handle_text] 處理文字時發生錯誤：{e}", exc_info=True)
        socketio.emit('status', '❌ 指令處理失敗')
        # 錯誤後恢復前端狀態
        with is_active_lock:
            current_status = "🟢 已啟動，請說指令..." if is_active else "⏳ 系統待機中，請說啟動詞..."
            socketio.emit('status', current_status)
            expression = '/static/animations/listening.gif' if is_active else '/static/animations/idle.gif'
            socketio.emit('expression', expression)


# --- Flask 路由 ---
@app.route('/')
def index():
    """提供前端 HTML 頁面。"""
    # 確保在請求上下文中運行，雖然對於 render_template_string 通常不是必須的
    with app.app_context():
        return render_template_string(HTML)

# ⭐ 修改 get_audio 路由名稱，與 handle_text 中 url_for 調用一致 ⭐
#    並使用 pathlib 進行安全路徑處理，與 delete_audio 類似
@app.route('/history_result/<filename>')
def get_audio(filename):
    """安全地提供 history_result 目錄下的音訊檔案。"""
    try:
        base_dir = Path('history_result').resolve()
        path_to_serve = (base_dir / filename).resolve()

        # 檢查：檔案是否存在 且 檔案的父目錄是 history_result 目錄
        if path_to_serve.is_file() and path_to_serve.parent == base_dir:
             # 使用 send_from_directory 安全地提供檔案
            return send_from_directory(base_dir, filename)
        else:
            logger.warning(f"[get_audio] 嘗試訪問無效或不安全的檔案路徑：{filename}")
            return "File not found", 404 # 返回 404 避免洩露信息
    except Exception as e:
        logger.error(f"[get_audio] 提供檔案 '{filename}' 失敗：{e}", exc_info=True)
        return "Error serving file", 500


# --- 主程式入口 ---
if __name__ == '__main__':
    # 創建 history_result 和 temp_audio 目錄
    Path('history_result').mkdir(exist_ok=True)
    Path('temp_audio').mkdir(exist_ok=True)
    logger.info("✅ 歷史紀錄和臨時檔案目錄已準備。")

    # ⭐ 重要：要運行包含 asyncio 的 SocketIO 應用，並且在 SocketIO 的同步 handler 中
    #    通過 run_coroutine_threadsafe 提交任務給異步 loop，你需要確保有一個 asyncio loop
    #    正在運行，並且 SocketIO 是以兼容異步的方式運行的。
    #    一種常見的方法是使用 eventlet 或 gevent 進行猴子補丁：
    #    import eventlet
    #    eventlet.monkey_patch()
    #    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True, eventlet=True)
    #
    #    如果你不使用猴子補丁，而只是標準的 threading，你需要自己啟動一個線程來運行 asyncio loop，
    #    並將該 loop 的實例傳遞給需要它的地方。這比猴子補丁更複雜。
    #    最簡單的測試方式是安裝 eventlet 並啟用猴子補丁。

    logger.info("🚀 啟動 SocketIO 伺服器...")
    # 請根據你的環境選擇合適的 SocketIO 運行方式
    # 如果已安裝 eventlet 並希望使用它，取消下面兩行的註解：
    # import eventlet
    # eventlet.monkey_patch()
    # socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True, eventlet=True)

    # 如果不使用 eventlet/gevent，則使用默認的 threading 模式運行
    # 注意：在默認 threading 模式下，run_coroutine_threadsafe 的行為可能需要根據 asyncio 版本和平台測試
    # 確保有一個 loop 在某處運行並可通過 asyncio.get_event_loop() 獲取。
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
