# live_transcriber/live_transcriber.py

import asyncio
import sounddevice
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# 初始化 Bedrock 客戶端（請先在環境變數或 AWS config 設定好憑證）
_bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
# Bedrock 分類 Prompt，回傳：START / STOP / INTERRUPT / COMMAND
_CLASSIFY_PROMPT = """
請判斷下列文字的意圖，僅回傳這四種之一（且僅該關鍵字）：
- START：啟動詞，“嘿，史多比”等相似詞
- STOP：結束詞，“再見史多比”等相似詞
- INTERRUPT：中斷指令，當當前指令仍在執行，卻又輸入新的命令，則分類為此類
- COMMAND：一般指令

文字："{text}"
"""

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

class LiveTranscriber:
    def __init__(self, region="us-west-2", callback=None, silence_timeout=2.0):
        self.client = TranscribeStreamingClient(region=region)
        self.callback = callback                # 傳入 main.py 的 handle_text
        self.silence_timeout = silence_timeout  # 停頓秒數
        self.buffer = []                        # 暫存文字
        self.timer_task = None                  # 靜音計時器
        self.active = False                     # 是否已在「啟動模式」
        self.current_task: asyncio.Task = None  # 正在執行的 callback

    async def classify_intent(self, text: str) -> str:
        """呼叫 Bedrock 判斷意圖：START/STOP/INTERRUPT/COMMAND"""
        prompt = _CLASSIFY_PROMPT.format(text=text.replace('"','\\"'))
        resp = _bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="text/plain",
            accept="application/json",
            inputStream=prompt.encode()
        )
        return resp["body"].read().decode().strip()

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        q = asyncio.Queue()

        def audio_cb(indata, fc, ti, status):
            loop.call_soon_threadsafe(q.put_nowait, (bytes(indata), status))

        with sounddevice.RawInputStream(
            channels=1, samplerate=16000, callback=audio_cb,
            blocksize=1024*2, dtype="int16"
        ):
            while True:
                yield await q.get()

    async def write_chunks(self, stream):
        async for chunk, _ in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    def _cancel_current(self):
        """中斷目前 callback"""
        if self.current_task and not self.current_task.done():
            print("中斷先前指令")
            self.current_task.cancel()
            self.current_task = None

    async def start(self):
        stream = await self.client.start_stream_transcription(
            language_code="zh-TW",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        handler = TranscribeHandler(stream.output_stream)
        tasks = asyncio.gather(
            self.write_chunks(stream),
            handler.handle_events()
        )
        print("系統等待啟動詞...")

        try:
            while True:
                text = await handler.final_transcripts.get()
                print("收到：", text)
                self.buffer.append(text)

                if self.timer_task:
                    self.timer_task.cancel()
                self.timer_task = asyncio.create_task(self._start_silence_timer())
        except asyncio.CancelledError:
            pass
        finally:
            tasks.cancel()
            await asyncio.gather(tasks, return_exceptions=True)

    async def _start_silence_timer(self):
        try:
            await asyncio.sleep(self.silence_timeout)
            await self.flush_buffer()
        except asyncio.CancelledError:
            pass

    async def flush_buffer(self):
        if not self.buffer:
            return
        text = " ".join(self.buffer).strip()
        self.buffer.clear()
        print("⏸ 停頓送出：", text)

        # 1. 先分類意圖
        intent = await self.classify_intent(text)
        print("分類結果：", intent)

        # 2. START：進入啟動模式
        if intent == "START":
            self.active = True
            print("🚀 進入啟動模式")
            return

        # 3. STOP：退出啟動模式並中斷指令
        if intent == "STOP":
            self.active = False
            print("停止所有動作")
            self._cancel_current()
            return

        # 4. INTERRUPT：中斷上一次 callback（保持 active）
        if intent == "INTERRUPT" and self.active:
            self._cancel_current()
            print("已中斷並等待新命令")
            return

        # 5. COMMAND：一般命令，在 active 狀態下才執行
        if intent == "COMMAND" and self.active:
            print("執行命令：", text)
            self._cancel_current()
            # asyncio.create_task 回傳一個 Task，未完成前可中斷
            self.current_task = asyncio.create_task(self.callback(text))
            return

        # 6. 其他情況：忽略
        print("❔ 未在啟動模式或非有效命令，忽略：", text)
