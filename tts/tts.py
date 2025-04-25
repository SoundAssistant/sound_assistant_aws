import boto3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.client_utils import get_polly_client

class Polly:
    def __init__(self):
        self.client = get_polly_client("polly")
        self.defaults = {
            "Engine": "neural", # 此為最接近真人的引擎
            "LanguageCode": "cmn-CN",   # zh-TW 目前無支援台灣腔調 官網只有這大陸腔跟香港腔
            "VoiceId": "Zhiyu",        #TianTian
            "OutputFormat": "mp3",
            "SampleRate": "16000", #這個參數是紀錄每秒紀錄多少個音點 越高品質越好 
        }

    def synthesize(self, text, output_filename):
        params = {**self.defaults, "Text": text}
        response = self.client.synthesize_speech(**params)

        with open(output_filename, "wb") as file:
            file.write(response["AudioStream"].read())

        print(f"{output_filename} saved successfully.")

# example
if __name__ == "__main__":
    polly = Polly()
    polly.synthesize("哈囉 我們是我要進外商", "./history_result/output.mp3")
