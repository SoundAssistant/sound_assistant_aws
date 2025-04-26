import streamlit as st
import asyncio
import time
import os
from pathlib import Path
from pydub import AudioSegment
import threading

from tools.flow_utils import task_flow, chat_flow, search_flow, action_flow
from live_transcriber.live_transcriber import LiveTranscriber

# --- Streamlit 頁面設定 ---
st.set_page_config(page_title="Robot Emotions", page_icon="🤖", layout="centered")
st.title("Robot Emotions 🤖")

# --- 預設畫面 ---
expression_placeholder = st.empty()
expression_placeholder.image("animations/wakeup.svg", use_column_width=True)

# --- 狀態記憶 ---
if "recording" not in st.session_state:
    st.session_state.recording = False

def set_expression(img_path):
    expression_placeholder.image(img_path, use_column_width=True)

async def process_text(text: str):
    set_expression('animations/thinking.gif')
    await asyncio.sleep(0.1)

    task_type = task_flow(text)
    mp3_path = None

    if task_type == "聊天":
        mp3_path = chat_flow(text)
    elif task_type == "查詢":
        mp3_path = search_flow(text)
    elif task_type == "行動":
        action_flow(text)
        mp3_path = None
    else:
        print(f"❓ 未知任務類型：{task_type}")

    # --- 撥放 mp3 ---
    if mp3_path and Path(mp3_path).exists():
        await asyncio.sleep(1)  # 等 mp3 產生
        set_expression('animations/speaking.gif')
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export("temp.wav", format="wav")
        audio_file = open("temp.wav", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        audio_file.close()

    set_expression('animations/idle.gif')

async def stt_flow_streamlit():
    """這是專門給 Streamlit 的版本"""
    async def handle_text(text: str):
        print(f"🎤 偵測到文字：{text}")
        await process_text(text)

    transcriber = LiveTranscriber(region="us-west-2", callback=handle_text)
    await transcriber.start()

def start_listening():
    async def runner():
        set_expression('animations/thinking.gif')
        await stt_flow_streamlit()

    threading.Thread(target=lambda: asyncio.run(runner())).start()

# --- 按鈕觸發 ---
if st.button("🎤 開始對話") and not st.session_state.recording:
    st.session_state.recording = True
    start_listening()
