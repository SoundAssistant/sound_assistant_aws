# -*- coding: utf-8 -*-
import socketio
import base64
import json
import boto3
import threading
import wave
import pyaudio
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
import os
load_dotenv() 
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
# ========== AWS Transcribe Streaming Client 初始化 ==========
transcribe_client = boto3.client(
    'transcribe',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name="us-east-1"
)

# ========== WebSocket 伺服器設定 ==========
sio = socketio.Client()

# ========== 語音錄製設定 ==========
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

import time

def poll_transcription_result(job_name):
    while True:
        result = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = result['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
            print("✅ Transcription 完成，結果在：", transcript_uri)
            break
        elif status == 'FAILED':
            print("❌ Transcription 失敗")
            break
        else:
            print("⌛ 等待 Transcription 完成...")
            time.sleep(5)
# ========== AWS Transcribe 任務提交（需開啟音訊串流服務） ==========
def transcribe_audio_stream(filename):
    job_name = "web-transcribe-job"
    try:
        with open(filename, 'rb') as f:
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': f's3://soundassistant/{filename}'},
                MediaFormat='wav',
                LanguageCode='zh-TW'
            )
        print("Transcribe Job Submitted.")
        poll_transcription_result(job_name)
    except Exception as e:
        print("Transcribe Error:", e)

# ========== 錄音後送出資料到 Server ==========
def record_and_send():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🔴 錄音中…")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 儲存音訊檔案
    wf = wave.open("temp.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 上傳到 S3 並觸發 Transcribe 任務
    upload_and_transcribe("temp.wav")

# ========== 上傳 S3 並觸發 ==========
def upload_and_transcribe(filename):
    try:
        s3 = boto3.client('s3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name="us-east-1")

        s3.upload_file(filename, "soundassistant", filename)
        print("📤 已上傳至 S3")
        transcribe_audio_stream(filename)

    except Exception as e:
        print("❌ 上傳或 Transcribe 發生錯誤：", e)

# ========== Socket.io 連線與錄音觸發 ==========
@sio.event
def connect():
    print("✅ 已連線前端 Socket")
    threading.Thread(target=record_and_send).start()

@sio.event
def disconnect():
    print("❌ 與前端 Socket 斷線")

if __name__ == "__main__":
    # sio.connect("http://localhost:5000")  # 根據你 Web UI 的實際 socket server 調整
    # sio.wait()
    record_and_send()  # 直接錄音並送出
