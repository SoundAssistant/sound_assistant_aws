import asyncio
import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


import asyncio
import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.final_transcripts = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:  # ✅ 只拿最終版
                for alt in result.alternatives:
                    text = alt.transcript.strip()
                    if text:
                        self.final_transcripts.append(text)
                        print(f"{text}")  # 這裡你可以替換成送 LLM、翻譯等功能


class LiveTranscriber:
    def __init__(self, region="us-west-2"):
        self.client = TranscribeStreamingClient(region=region)

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
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
            language_code="zh-TW",  # ✅ 這個一定要給！
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        handler = TranscribeHandler(stream.output_stream)
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())



if __name__ == "__main__":
    transcriber = LiveTranscriber(region="us-west-2")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(transcriber.start())
    except KeyboardInterrupt:
        print("\n🎤 偵測結束。")
    finally:
        loop.close()