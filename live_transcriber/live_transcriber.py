# live_transcriber/live_transcriber.py
# -----------------------------------------------------------
# 以 AWS Transcribe Streaming + Bedrock Claude-3 Haiku 實現：
#   • 一直錄音、停頓後送文字
#   • Bedrock 分類 START / STOP / INTERRUPT / COMMAND
#   • 中斷、啟動、結束控制
# -----------------------------------------------------------

import asyncio, json, sounddevice, boto3
from botocore.config import Config
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# ---------- Bedrock 參數 ----------
REGION    = "us-west-2"                                   # 改成你的區域
MODEL_ID  = "anthropic.claude-3-haiku-20240307-v1:0"       # 改成你的模型
BEDROCK   = boto3.client("bedrock-runtime",
                         region_name=REGION,
                         config=Config(read_timeout=20, connect_timeout=5))
# ----------------------------------

# ---------- 分類提示 ----------
_CLASSIFY_PROMPT = """
請判斷下列文字的意圖，只能回答以下四個字串之一：
START / STOP / INTERRUPT / COMMAND

文字：「{text}」
"""
# ----------------------------------

# ---------- Transcribe 事件處理 ----------
class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.final_transcripts = asyncio.Queue()

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for r in transcript_event.transcript.results:
            if not r.is_partial:
                for alt in r.alternatives:
                    t = alt.transcript.strip()
                    if t:
                        await self.final_transcripts.put(t)
# ------------------------------------------

class LiveTranscriber:
    def __init__(self, region="us-west-2", callback=None, silence_timeout=2.0):
        self.client = TranscribeStreamingClient(region=region)
        self.callback = callback
        self.silence_timeout = silence_timeout

        self.buffer: list[str] = []
        self.timer_task: asyncio.Task | None = None
        self.active = False
        self.current_task: asyncio.Task | None = None

    # ---------- Bedrock 分類 ----------
    async def classify_intent(self, text: str) -> str:
        user_prompt = _CLASSIFY_PROMPT.format(text=text.replace('"', '\\"'))
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1,
            "temperature": 0,
            "messages": [
                {"role": "system",
                 "content": "你是語音助理的意圖分類器，只回答 START、STOP、INTERRUPT、COMMAND 四種之一。"},
                {"role": "user", "content": user_prompt}
            ]
        }

        def _invoke():
            resp = BEDROCK.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8")
            )
            data = json.loads(resp["body"].read())
            return data["content"][0]["text"].strip().upper()

        try:
            intent = await asyncio.to_thread(_invoke)
            return intent if intent in {"START", "STOP", "INTERRUPT", "COMMAND"} else "IGNORE"
        except Exception as e:
            print("⚠️  Bedrock 失敗：", e)
            return "IGNORE"
    # -----------------------------------

    # ---------- 麥克風 ----------
    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _cb(indata, *_):
            loop.call_soon_threadsafe(q.put_nowait, (bytes(indata), None))

        with sounddevice.RawInputStream(
            samplerate=16000, channels=1, dtype="int16",
            callback=_cb, blocksize=2048
        ):
            while True:
                yield await q.get()
    # -----------------------------------

    async def write_chunks(self, stream):
        async for chunk, _ in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    # ---------- 取消目前任務 ----------
    def _cancel_current(self):
        if self.current_task and not self.current_task.done():
            print("⚡ 中斷上一個命令")
            self.current_task.cancel()
        self.current_task = None
    # -----------------------------------

    # ========== 主流程 ==========
    async def start(self):
        stream = await self.client.start_stream_transcription(
            language_code="zh-TW",
            media_sample_rate_hz=16000,
            media_encoding="pcm"
        )
        handler = TranscribeHandler(stream.output_stream)
        runners = asyncio.gather(self.write_chunks(stream), handler.handle_events())

        print("🎤 系統待機：請說啟動詞…")

        try:
            while True:
                t = await handler.final_transcripts.get()
                print("📝 收到片段：", t)
                self.buffer.append(t)

                if self.timer_task:
                    self.timer_task.cancel()
                self.timer_task = asyncio.create_task(self._start_silence_timer())
        except asyncio.CancelledError:
            pass
        finally:
            runners.cancel()
            await asyncio.gather(runners, return_exceptions=True)
    # ===========================

    async def _start_silence_timer(self):
        try:
            await asyncio.sleep(self.silence_timeout)
            await self.flush_buffer()
        except asyncio.CancelledError:
            pass

    # ---------- 停頓後送出 ----------
    async def flush_buffer(self):
        if not self.buffer:
            return

        text = " ".join(self.buffer).strip()
        self.buffer.clear()
        print("⏸ 停頓結束，傳送文字：", text)

        intent = await self.classify_intent(text)
        print("🔍 意圖分類：", intent)

        # START
        if intent == "START":
            self.active = True
            print("🚀 已啟動，等待命令…")
            return

        # STOP
        if intent == "STOP":
            if self.active:
                self._cancel_current()
                print("🛑 結束詞收到，回到待機")
            self.active = False
            return

        # INTERRUPT
        if intent == "INTERRUPT":
            if self.active and self.current_task and not self.current_task.done():
                self._cancel_current()
                print("🔄 已中斷，等待新命令…")
            else:
                print("ℹ️ Interrupt 但無任務可中斷")
            return

        # COMMAND
        if intent == "COMMAND":
            if not self.active:
                print("❔ 尚未啟動，忽略命令")
                return
            if self.current_task and not self.current_task.done():
                self._cancel_current()
            print("✅ 執行新命令：", text)
            self.current_task = asyncio.create_task(self.callback(text))
            return

        print("ℹ️ 非法或忽略意圖")
# -----------------------------------------------------------
