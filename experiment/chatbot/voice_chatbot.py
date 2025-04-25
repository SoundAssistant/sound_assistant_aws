import os
import json
import asyncio

import pyaudio
import boto3

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import AudioEvent
from amazon_transcribe.handlers import TranscriptResultStreamHandler

# ---------- 設定 ----------
REGION = os.getenv("AWS_REGION", "us-east-1")
INPUT_RATE = 16000
CHANNELS = 1
CHUNK = 1024
VOICE_ID = os.getenv("VOICE_ID", "Zhiyu")
CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# ---------- AWS 客戶端 ----------
transcribe_client = TranscribeStreamingClient(region=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
polly = boto3.client("polly", region_name=REGION)

# ---------- 麥克風串流 generator ----------
async def mic_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=INPUT_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            yield AudioEvent(audio_chunk=data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# ---------- 處理 Transcribe 回傳的文字 ----------
class MyTranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream, output_callback):
        super().__init__(transcript_result_stream)
        self.output_callback = output_callback

    async def handle_transcript_event(self, transcript_event):
        for result in transcript_event.transcript.results:
            if result.is_partial:
                continue
            if result.alternatives:
                text = result.alternatives[0].transcript.strip()
                if text:
                    await self.output_callback(text)

# ---------- 呼叫 Claude 回應並回傳中文 ----------
def call_claude_respond_zh(text):
    body = {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    resp = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode('utf-8')
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"].strip()

# ---------- 透過 Polly 合成並播放 ----------
def speak(text: str):
    resp = polly.synthesize_speech(
        Text=text,
        OutputFormat="pcm",
        VoiceId=VOICE_ID,
        SampleRate=str(INPUT_RATE)
    )
    audio_stream = resp["AudioStream"]
    pa = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16,
                  channels=1,
                  rate=INPUT_RATE,
                  output=True)
    try:
        while True:
            chunk = audio_stream.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
    finally:
        out.stop_stream()
        out.close()
        pa.terminate()

# ---------- 當接收到最終文字的 callback ----------
async def handle_text(text: str):
    print(f"🗣️ 你說：{text}")
    reply = call_claude_respond_zh(text)
    print(f"🤖 Claude 回應：{reply}")
    speak(reply)

# ---------- 主流程 ----------


from amazon_transcribe.model import StartStreamTranscriptionRequest

async def main():
    # 建立 StartStreamTranscriptionRequest
    request = StartStreamTranscriptionRequest(
        language_code="en-US",
        media_sample_rate_hz=INPUT_RATE,
        media_encoding="pcm"
    )

    # 建立音訊串流
    audio_stream = mic_stream()

    # 建立 streaming_request
    streaming_request = transcribe_client.streaming_request(
        request=request,
        audio_stream=audio_stream
    )

    # 啟動 Transcribe 串流
    stream = await transcribe_client.start_stream_transcription(streaming_request)

    # 建立並啟動 handler
    handler = MyTranscribeHandler(
        stream.transcript_result_stream,
        handle_text
    )
    await handler.handle_events()




if __name__ == "__main__":
    asyncio.run(main())
