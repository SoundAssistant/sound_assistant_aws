import boto3
import sys
import os
import io
from pydub import AudioSegment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.client_utils import get_polly_client

class PollyTTS:
    def __init__(self):
        self.client = get_polly_client("polly")
        self.defaults = {
            "Engine": "neural",
            "LanguageCode": "cmn-CN",
            "VoiceId": "Zhiyu",
            "OutputFormat": "mp3",     # Polly 只能生 mp3, ogg_vorbis, pcm
            "SampleRate": "16000",
        }

    def synthesize(self, text, output_filename):
        params = {**self.defaults, "Text": text}
        response = self.client.synthesize_speech(**params)
        audio_stream = response["AudioStream"].read()

        if output_filename.endswith(".mp3"):
            # 直接存 mp3
            with open(output_filename, "wb") as file:
                file.write(audio_stream)
            print(f"{output_filename} saved as MP3 successfully.")

        elif output_filename.endswith(".wav"):
            # 轉成 wav 再存
            audio = AudioSegment.from_file(io.BytesIO(audio_stream), format="mp3")
            audio.export(output_filename, format="wav")
            print(f"{output_filename} saved as WAV successfully.")

        else:
            raise ValueError("Output filename must end with .mp3 or .wav")

# --- example ---
if __name__ == "__main__":
    polly = PollyTTS()
    polly.synthesize("哈囉，我們是我要進外商", "./history_result/output.wav")   # 存成 wav
    polly.synthesize("哈囉，我們是我要進外商", "./history_result/output.mp3")   # 存成 mp3
