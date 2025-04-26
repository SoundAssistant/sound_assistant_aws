import asyncio
import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.final_transcripts = asyncio.Queue()

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    text = alt.transcript.strip()
                    if text:
                        await self.final_transcripts.put(text)

class LiveTranscriber:
    def __init__(self, region="us-west-2", callback=None, silence_timeout=3.5):
        self.client = TranscribeStreamingClient(region=region)
        self.callback = callback
        self.silence_timeout = silence_timeout
        self.buffer = []
        self.timer_task = None

        # 🔥 自動找一個有"mic"字樣的裝置
        devices = sounddevice.query_devices()
        mic_index = None
        for i, d in enumerate(devices):
            if 'mic' in d['name'].lower() and d['max_input_channels'] > 0:
                mic_index = i
                break

        if mic_index is None:
            print("⚠️ 找不到麥克風裝置，會用預設輸入裝置")
            self.input_device = None
        else:
            print(f"🎤 選用麥克風裝置：{devices[mic_index]['name']}")
            self.input_device = mic_index
    def is_valid_text(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False  # 空的不要
        if len(text) < 2:
            return False  # 太短的不要（像 "嗯"）
        if all(c in "，。？！、,.?! " for c in text):
            return False  # 全是標點符號的不要
        return True

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
            device=self.input_device,  # 🔥 指定麥克風
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=1024 * 2,
            dtype="int16",
        )

        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

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

        try:
            while True:
                text = await handler.final_transcripts.get()
                # ✅ 簡單噪音判斷
                if not self.is_valid_text(text):
                    print(f"⚡ 濾掉無效文字：'{text}'")
                    return  # 無效的就直接忽略，不加入 buffer

                print(f"📝 偵測到新文字：{text}")
                self.buffer.append(text)


                # 有新的文字，重新啟動 silence timer
                if self.timer_task:
                    self.timer_task.cancel()

                self.timer_task = asyncio.create_task(self._start_silence_timer())

        except asyncio.CancelledError:
            print("🛑 中斷偵測，清理資源...")
        finally:
            tasks.cancel()
            await asyncio.gather(tasks, return_exceptions=True)

    async def _start_silence_timer(self):
        try:
            await asyncio.sleep(self.silence_timeout)
            await self.flush_buffer()  # 時間到了就送出 buffer
        except asyncio.CancelledError:
            pass  # 被新的文字打斷就什麼都不做

    async def flush_buffer(self):
        if not self.buffer:
            return

        full_text = " ".join(self.buffer).strip()
        print(f"✅ 使用者停頓，送出整段文字：{full_text}")

        if self.callback:
            await self.callback(full_text)

        self.buffer.clear()

        # 🔥 強制休息 2~3秒
        wait_time = 3 + (asyncio.get_event_loop().time() % 1)  # 2.0~3.0秒之間
        print(f"⏳ 等待 {wait_time:.2f} 秒避免過快連續送出...")
        await asyncio.sleep(wait_time)

