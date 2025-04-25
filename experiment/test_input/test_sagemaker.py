import os, wave, base64, json
import boto3, pyaudio

# ---------- 共用設定 ----------
REGION = "us-east-1"
ASR_ENDPOINT = "your-whisper-endpoint"
TTS_ENDPOINT = "your-tts-endpoint"
BEDROCK_MODEL = "anthropic.claude-v2"   # 或其他支援文字翻譯的模型

# ---------- 初始化 Client ----------
sm = boto3.client("sagemaker-runtime", region_name=REGION)
br = boto3.client("bedrock-runtime",   region_name=REGION)

# ---------- 1. 錄音並回傳 PCM bytes ----------
def record_pcm(seconds=3):
    RATE, CHUNK = 16000, 1024
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(int(RATE/CHUNK*seconds))]
    stream.stop_stream(); stream.close(); pa.terminate()
    return b"".join(frames)

# ---------- 2. ASR：PCM → 文字 ----------
def asr_whisper(pcm_bytes):
    resp = sm.invoke_endpoint(
        EndpointName=ASR_ENDPOINT,
        ContentType="application/octet-stream",
        Body=pcm_bytes
    )
    text = resp["Body"].read().decode("utf-8")
    return text.strip()

# ---------- 3. 翻譯：英文 → 中文文字 ----------
def translate_bedrock(text_en):
    payload = {
      "inputs": text_en,
      "parameters": {"temperature":0.0, "max_tokens":512}
    }
    resp = br.invoke_model(
        modelId=BEDROCK_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload).encode("utf-8")
    )
    result = json.loads(resp["body"].read())
    # Claude 類模型回傳通常在 result["completion"]
    return result.get("completion", "").strip()

# ---------- 4. TTS：中文文字 → PCM ----------
def tts_sagemaker(text_zh):
    resp = sm.invoke_endpoint(
        EndpointName=TTS_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"text": text_zh}).encode("utf-8")
    )
    # 假設 TTS endpoint 回傳 raw PCM
    return resp["Body"].read()

# ---------- 5. 寫成 WAV 或播放 ----------
def save_wav(pcm_bytes, filename="out.wav", rate=24000):
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(pcm_bytes)
    wf.close()
    print(f"✅ 已儲存為 {filename}")

# ---------- 主流程 ----------
if __name__ == "__main__":
    print("🔴 開始錄音…")
    pcm = record_pcm(3)

    print("🔍 ASR 轉文字…")
    text = asr_whisper(pcm)
    print("→ 英文：", text)

    print("🌐 Bedrock 翻譯…")
    zh = translate_bedrock(text)
    print("→ 中文：", zh)

    print("🔊 TTS 生成…")
    pcm2 = tts_sagemaker(zh)

    save_wav(pcm2, "translation_output.wav", rate=24000)
