# live_transcriber/live_transcriber.py
# -----------------------------------------------------------
# 以 AWS Transcribe Streaming + Bedrock (Haiku) 實現：
#  1. 持續錄音（麥克風永不關閉）
#  2. 停頓後將累積文字送 Bedrock 分類
#  3. 判斷 START / STOP / INTERRUPT / COMMAND
#  4. 啟動模式下才能執行 callback；INTERRUPT 會中斷舊命令
# -----------------------------------------------------------

import asyncio
import sounddevice
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# ---------- AWS 連線 ----------
REGION = "us-west-2"          # ✅ 改成你的預設區域
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # ✅ 改成你有權限的 Bedrock 模型
_bedrock = boto3.client("bedrock-runtime", region_name=REGION)
# ------------------------------

_CLASSIFY_PROMPT = """
請判斷下列文字的意圖，僅回傳以下四個關鍵字之一（不含其他字）：
START   # 像「嘿史多比」啟動詞
STOP    # 像「史多比再見」結束詞
INTERRUPT  # 在啟動模式中，嘗試打斷上一個還在執行的命令
COMMAND # 一般命令

文字："{text}"
"""

# ========== Transcribe 事件處理 ==========
class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.final_transcripts = asyncio.Queue()

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    t = alt.transcript.strip()
                    if t:
                        await self.final_transcripts.put(t)
# ========================================

class LiveTranscriber:
    def __init__(self, region="us-west-2", callback=None, silence_timeout=2.0):
        self.client = TranscribeStreamingClient(region=region)
        self.callback = callback
        self.silence_timeout = silence_timeout

        self.buffer: list[str] = []           # 累積文字
        self.timer_task: asyncio.Task | None = None
        self.active = False                   # 啟動模式
        self.current_task: asyncio.Task | None = None

    # ---------- Bedrock 分類 ----------
    async def classify_intent(self, text: str) -> str:
        prompt = _CLASSIFY_PROMPT.format(text=text.replace('"', '\\"'))
        try:
            resp = _bedrock.invoke_model(
                modelId=MODEL_ID,
                contentType="text/plain",
                accept="application/json",
                inputStream=prompt.encode("utf-8")
            )
            intent = resp["body"].read().decode().strip().upper()
            return intent
        except Exception as e:
            print(f"Bedrock 失敗，預設 IGNORE：{e}")
            return "IGNORE"
    # -----------------------------------

    # ---------- 麥克風 ---------- 
    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _audio_cb(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(q.put_nowait, (bytes(indata), status))

        with sounddevice.RawInputStream(
            samplerate=16000, channels=1, dtype="int16",
            callback=_audio_cb, blocksize=1024 * 2
        ):
            while True:
                yield await q.get()
    # -----------------------------------

    async def write_chunks(self, stream):
        async for chunk, _ in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    # ---------- 任務取消 ----------
    def _cancel_current(self):
        if self.current_task and not self.current_task.done():
            print("⚡ 中斷上一個命令")
            self.current_task.cancel()
        self.current_task = None
    # -----------------------------------

    # ========== 主入口 ==========
    async def start(self):
        stream = await self.client.start_stream_transcription(
            language_code="zh-TW",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        handler = TranscribeHandler(stream.output_stream)

        runners = asyncio.gather(self.write_chunks(stream), handler.handle_events())
        print("系統待機：請說啟動詞…")

        try:
            while True:
                t = await handler.final_transcripts.get()
                print("收到片段：", t)
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

        # ---------- START ----------
        if intent == "START":
            self.active = True
            print("已啟動，等待命令…")
            return

        # ---------- STOP ----------
        if intent == "STOP":
            if self.active:
                self._cancel_current()
                print("結束詞收到，回到待機")
            self.active = False
            return

        # ---------- INTERRUPT ----------
        if intent == "INTERRUPT":
            if self.active and self.current_task and not self.current_task.done():
                self._cancel_current()
                print("已中斷，等待新命令…")
            else:
                print("ℹ️ Interrupt 但無任務可中斷，忽略")
            return

        # ---------- COMMAND ----------
        if intent == "COMMAND":
            if not self.active:
                print("❔ 尚未啟動，忽略命令")
                return

            # 若任務仍在進行，先中斷
            if self.current_task and not self.current_task.done():
                self._cancel_current()

            print("執行新命令：", text)
            self.current_task = asyncio.create_task(self.callback(text))
            return

        # ---------- 其他 ----------
        print("ℹ️ 非法或忽略意圖")
# -----------------------------------------------------------
