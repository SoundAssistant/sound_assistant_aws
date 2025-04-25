import os
import json
import asyncio
import pyaudio
import boto3
import logging # Import logging module

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import AudioEvent, StartStreamTranscriptionRequest
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from botocore.exceptions import BotoCoreError, ClientError

# ---------- Configuration ----------
REGION = os.getenv("AWS_REGION", "us-east-1")
INPUT_RATE = 16000
OUTPUT_RATE = 16000 # Polly default for pcm
CHANNELS = 1
CHUNK_SIZE = 1024 # Smaller buffer size might reduce latency slightly, but 1024 is standard
FORMAT = pyaudio.paInt16
VOICE_ID = os.getenv("VOICE_ID", "Zhiyu") # Use a Chinese voice for Chinese responses
CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # Using Haiku for potentially lower latency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- AWS Clients ----------
# Transcribe client is used directly in the streaming handler
transcribe_client = TranscribeStreamingClient(region=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
polly = boto3.client("polly", region_name=REGION)

# ---------- Global/Shared State (Managed by Assistant class) ----------
# We will manage state like audio queues and barge-in within the Assistant class

# ---------- Microphone Input Stream Generator ----------
async def mic_stream(chunk_size: int):
    """Generates audio chunks from the microphone."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=INPUT_RATE,
                    input=True,
                    frames_per_buffer=chunk_size)
    
    logger.info("Microphone input stream opened.")
    
    try:
        while True:
            try:
                # 讀取音頻數據
                data = stream.read(chunk_size, exception_on_overflow=False)
                if data:
                    # 創建音頻事件
                    audio_event = AudioEvent(
                        audio_chunk=data
                    )
                    yield audio_event
            except Exception as e:
                logger.error(f"Error reading from microphone stream: {e}")
                break
    finally:
        logger.info("Closing microphone input stream.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        logger.info("PyAudio terminated.")


# ---------- Transcribe Event Handler ----------
class MyTranscribeHandler(TranscriptResultStreamHandler):
    """Handles transcription results and sends final text to the assistant."""
    def __init__(self, transcript_result_stream, assistant_instance):
        super().__init__(transcript_result_stream)
        self.assistant = assistant_instance
        logger.info("Transcribe handler initialized.")

    async def handle_transcript_event(self, transcript_event):
        # logger.debug("Received transcript event")
        for result in transcript_event.transcript.results:
            # We only process final results for sending to Bedrock
            if result.is_partial:
                # logger.debug(f"Partial transcript: {result.alternatives[0].transcript.strip()}")
                continue
            if result.alternatives:
                text = result.alternatives[0].transcript.strip()
                if text:
                    logger.info(f"🗣️ 你說：{text}")
                    # Put the final text into the assistant's queue for processing
                    await self.assistant.transcript_queue.put(text)

# ---------- Assistant Orchestrator ----------
class VoiceAssistant:
    """Orchestrates Transcribe, Bedrock, Polly, and audio playback."""
    def __init__(self):
        self.transcript_queue = asyncio.Queue() # Queue for final transcripts from Transcribe
        self.audio_output_queue = asyncio.Queue() # Queue for audio chunks from Polly
        self.stop_event = asyncio.Event() # Event to signal shutdown
        self.pyaudio_instance = None
        self.output_stream = None
        self.current_polly_task = None # Reference to the currently running Polly synthesis task
        self.is_assistant_speaking = False # Flag to indicate if assistant audio is playing

    async def start(self):
        """Starts all necessary tasks for the assistant."""
        logger.info("Starting voice assistant.")
        self.pyaudio_instance = pyaudio.PyAudio()
        self.output_stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_RATE, # Use the rate that Polly synthesizes at
            output=True,
            frames_per_buffer=CHUNK_SIZE # Match input chunk size for simplicity
        )
        logger.info("PyAudio output stream opened.")

        # Create and start tasks
        # Task to handle transcription
        self.transcribe_task = asyncio.create_task(self._run_transcription())
        # Task to process transcripts and interact with Bedrock/Polly
        self.pipeline_task = asyncio.create_task(self._run_bedrock_polly_pipeline())
        # Task to play audio from the output queue
        self.playback_task = asyncio.create_task(self._run_audio_playback())

        logger.info("Voice assistant started. Speak to begin.")

    async def stop(self):
        """Signals shutdown and cleans up resources."""
        logger.info("Stopping voice assistant.")
        self.stop_event.set() # Signal tasks to stop

        # Cancel tasks and wait for them
        tasks_to_cancel = [
            self.transcribe_task,
            self.pipeline_task,
            self.playback_task
        ]
        if self.current_polly_task and not self.current_polly_task.done():
             tasks_to_cancel.append(self.current_polly_task)

        for task in tasks_to_cancel:
            if task and not task.done():
                 task.cancel()
                 try:
                     await task # Wait for cancellation to complete
                 except asyncio.CancelledError:
                     pass # Expected exception

        # Clear queues to unblock any waiting tasks before closing streams
        while not self.transcript_queue.empty():
            try: self.transcript_queue.get_nowait()
            except asyncio.QueueEmpty: break
        while not self.audio_output_queue.empty():
            try: self.audio_output_queue.get_nowait()
            except asyncio.QueueEmpty: break

        # Close streams and PyAudio
        if self.output_stream:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
            logger.info("PyAudio output stream closed.")
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            logger.info("PyAudio terminated.")

        logger.info("Voice assistant stopped.")


    # 修改 _run_transcription 方法中的相關代碼
    async def _run_transcription(self):
        logger.info("Transcription task started.")
        while not self.stop_event.is_set():
            try:
                # 創建音頻流
                audio_generator = mic_stream(CHUNK_SIZE)
                
                # 開始串流轉錄
                async with transcribe_client as client:
                    stream = await client.start_stream_transcription(
                        language_code="en-US",
                        media_sample_rate_hz=INPUT_RATE,
                        media_encoding="pcm"
                    )
                    
                    logger.info("Transcribe stream connected.")
                    
                    # 發送音頻數據
                    async for audio_chunk in audio_generator:
                        await stream.input_stream.send_audio_event(audio_chunk)
                    
                    # 發送結束信號
                    await stream.input_stream.end_stream()
                    
                    # 處理轉錄結果
                    handler = MyTranscribeHandler(
                        stream.output_stream,
                        self
                    )
                    
                    await handler.handle_events()
                    
                logger.info("Transcribe stream finished or disconnected.")

            except asyncio.CancelledError:
                logger.info("Transcribe task cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in Transcribe task: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(2)



    async def _run_bedrock_polly_pipeline(self):
        """Processes transcripts, interacts with Bedrock, and queues audio from Polly."""
        logger.info("Bedrock/Polly pipeline task started.")
        loop = asyncio.get_event_loop()

        while not self.stop_event.is_set():
            try:
                # Wait for a new final transcript
                user_text = await self.transcript_queue.get()
                logger.debug(f"Received text from queue: {user_text}")

                # === Barge-in Handling ===
                # If assistant was speaking, receiving new user text means barge-in
                if self.is_assistant_speaking:
                    logger.info("Barge-in detected. Interrupting assistant speech.")
                    # Cancel the currently running Polly synthesis task
                    if self.current_polly_task and not self.current_polly_task.done():
                        logger.debug("Cancelling current Polly task.")
                        self.current_polly_task.cancel()
                        # Give it a moment to clean up (especially emptying queue)
                        await asyncio.sleep(0.05)
                    # Clear the audio output queue immediately
                    while not self.audio_output_queue.empty():
                        try: self.audio_output_queue.get_nowait()
                        except asyncio.QueueEmpty: break
                    logger.debug("Audio output queue cleared.")
                    self.is_assistant_speaking = False # Reset flag

                # === Interact with Bedrock (Streaming Text Out) ===
                logger.info(f"🤖 Calling Claude (streaming) with: '{user_text}'")
                claude_body = {
                    "messages": [{"role": "user", "content": user_text}],
                    "max_tokens": 1024,
                    "temperature": 0.7
                }
                claude_response_stream = await loop.run_in_executor(
                    None, # Use default thread pool
                    lambda: bedrock.invoke_model_with_response_stream(
                        modelId=CLAUDE_MODEL_ID,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(claude_body).encode('utf-8')
                    )['body'] # Get the streaming body
                )

                assistant_response_text = ""
                logger.info("🤖 Claude Response (streaming):")

                # Process the Bedrock streaming text response
                # We accumulate the entire response before sending to Polly for simplicity
                # For true low-latency, you'd chunk by sentence/phrase and call Polly multiple times concurrently
                async for event in claude_response_stream:
                    if event.get('chunk'):
                         chunk_data = json.loads(event['chunk']['bytes'].decode('utf-8'))
                         if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                              text_chunk = chunk_data['delta']['text']
                              # print(text_chunk, end='', flush=True) # Print chunk as it arrives
                              assistant_response_text += text_chunk
                    # Check for stop signal periodically during streaming
                    if self.stop_event.is_set():
                        logger.warning("Stop event set during Bedrock streaming, exiting.")
                        break # Exit the async for loop

                # print("", flush=True) # Add newline after full response
                logger.info(f"🤖 Full Claude response received ({len(assistant_response_text)} chars).")

                if not assistant_response_text or self.stop_event.is_set():
                     logger.warning("Empty or interrupted Claude response, skipping TTS.")
                     continue # Skip TTS if no text or stopping

                # === Call Polly (Non-streaming request, Streaming Audio Out) ===
                logger.info("✨ Synthesizing speech with Polly...")
                # Create a new task for Polly synthesis so we don't block the pipeline task
                # Store the task reference for barge-in cancellation
                self.current_polly_task = asyncio.create_task(
                    self._synthesize_and_queue_audio(assistant_response_text)
                )
                # Wait for the synthesis and queuing to finish for this turn
                await self.current_polly_task

            except asyncio.CancelledError:
                 logger.info("Pipeline task cancelled.")
                 break # Exit loop on cancellation
            except BotoCoreError as e:
                logger.error(f"Bedrock or Polly BotoCoreError: {e}")
                # Decide how to handle errors (retry, log, etc.)
                await asyncio.sleep(1) # Wait a bit before processing next transcript
            except ClientError as e:
                logger.error(f"Bedrock or Polly ClientError: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error in pipeline task: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1) # Wait a bit

        logger.info("Bedrock/Polly pipeline task stopped.")


    async def _synthesize_and_queue_audio(self, text_to_speak: str):
        """Calls Polly and puts the resulting audio chunks into the output queue."""
        logger.debug(f"Starting Polly synthesis for text: '{text_to_speak[:50]}...'")
        loop = asyncio.get_event_loop()
        self.is_assistant_speaking = True # Set speaking flag

        try:
            # Call Polly's synthesize_speech (blocking I/O, run in executor)
            polly_response = await loop.run_in_executor(
                None, # Use default thread pool
                lambda: polly.synthesize_speech(
                    Text=text_to_speak,
                    OutputFormat="pcm",
                    VoiceId=VOICE_ID,
                    SampleRate=str(OUTPUT_RATE)
                )
            )

            audio_stream = polly_response["AudioStream"]
            logger.debug("Polly AudioStream received.")

            # Read from the audio stream and put chunks into the queue
            # Read in a separate thread to avoid blocking asyncio loop
            def read_audio_chunks():
                 try:
                    while True:
                         # Check if stop event is set or task is cancelled
                         if self.stop_event.is_set() or self.current_polly_task.cancelled():
                             logger.debug("Polly synthesis task cancelled/stopping.")
                             break
                         # Read a chunk (blocking)
                         chunk = audio_stream.read(CHUNK_SIZE)
                         if not chunk:
                             logger.debug("Polly audio stream finished.")
                             break
                         # Put chunk into the async queue (non-blocking)
                         asyncio.run_coroutine_threadsafe(
                             self.audio_output_queue.put(chunk), loop
                         )
                         # Small sleep to yield and allow cancellation check
                         time.sleep(0.001) # Use time.sleep for blocking reads

                 except Exception as e:
                    logger.error(f"Error reading from Polly audio stream: {e}")
                 finally:
                    if hasattr(audio_stream, 'close'):
                        audio_stream.close() # Ensure stream is closed

            # Run the blocking read loop in a thread
            await loop.run_in_executor(None, read_audio_chunks)

        except asyncio.CancelledError:
            logger.info("Polly synthesis task cancelled.")
            # The read_audio_chunks executor task should handle cancellation internally
            # or be cancelled externally if needed, but checking self.current_polly_task.cancelled() inside helps.
        except Exception as e:
            logger.error(f"Error during Polly synthesis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure the speaking flag is reset
            self.is_assistant_speaking = False
            logger.debug("Polly synthesis task finished/cancelled.")
            # The playback task will continue until the queue is empty

    async def _run_audio_playback(self):
        """Reads audio chunks from the queue and plays them."""
        logger.info("Audio playback task started.")
        loop = asyncio.get_event_loop()

        while not self.stop_event.is_set():
            try:
                # Wait for an audio chunk
                audio_chunk = await self.audio_output_queue.get()
                logger.debug(f"Playing audio chunk of size {len(audio_chunk)} bytes.")

                if audio_chunk:
                    # Write to the output stream (blocking operation, run in executor)
                    await loop.run_in_executor(
                        None, # Use default thread pool
                        self.output_stream.write,
                        audio_chunk
                    )
                self.audio_output_queue.task_done() # Mark the item as processed

            except asyncio.CancelledError:
                 logger.info("Audio playback task cancelled.")
                 # Clearing queue is handled in stop() or pipeline task on barge-in
                 break # Exit loop on cancellation
            except Exception as e:
                logger.error(f"Error during audio playback: {e}")
                import traceback
                traceback.print_exc()
                # Continue trying to play subsequent chunks

        logger.info("Audio playback task stopped.")


# ---------- Main Execution ----------
async def main():
    """Main function to run the assistant."""
    assistant = VoiceAssistant()

    # Start the assistant and wait for the stop event
    try:
        await assistant.start()
        # Keep the main task alive until interrupted
        await assistant.stop_event.wait() # Wait until stop_event is set
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping.")
    finally:
        # Ensure stop is called for cleanup
        await assistant.stop()

if __name__ == "__main__":
    # Set your AWS credentials here or use environment variables
    # os.environ['AWS_ACCESS_KEY_ID'] = "YOUR_ACCESS_KEY_ID"
    # os.environ['AWS_SECRET_ACCESS_KEY'] = "YOUR_SECRET_ACCESS_KEY"
    # os.environ['AWS_DEFAULT_REGION'] = "us-east-1" # Or your region

    # Check if AWS credentials are set (basic check)
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
         logger.warning("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set. Ensure credentials are configured.")
         # You might want to exit here or use a credentials resolver that looks elsewhere

    # Run the main async function
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Application encountered a top-level error: {e}")
        import traceback
        traceback.print_exc()