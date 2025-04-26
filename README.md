# SoundAssistant

## 專案流程以及內容


## Live展示:
[影片連結](url)

## 專案架構: 
![image](https://github.com/SoundAssistant/sound_assistant_aws/blob/main/arch.png)

**Environment**<br>
Platform: Windows 11<br>
Python version: 3.12.9<br>

**Configuration**<br>
You need to add those settings in `config.env` your `HACK_MD_API_TOKEN`

**Build**<br>
For conda virtual environment (recommanded)
Pre-requirement : You need to create the tavily and aws access token in `.env`
```
conda create --name 'YOUR_ENV_NAME'
conda activate 'YOUR_ENV_NAME'
pip install -r requirements.txt
```
For python virtual environment
```
python -m venv 'YOUR_ENV_NAME'
source 'YOUR_ENV_NAME'/bin/activate
pip install -r requirements.txt
```
**Run**<br>
```
streamlit run app.py
```
After running this command, the server will listen on port `8501`.

## Trello
[Trello](https://trello.com/b/jRL2x0qk/aws)<br>

## Reference
[Polly](https://docs.aws.amazon.com/zh_tw/polly/latest/dg/bilingual-voices.html)<br>
[Transcribe](https://aws.amazon.com/tw/transcribe/)<br>
[Bedrock](https://aws.amazon.com/tw/bedrock/)<br>
[SageMaker](https://aws.amazon.com/tw/sagemaker/?trk=346c6f6e-fbca-42ed-9c22-666d71fff455&sc_channel=ps&ef_id=Cj0KCQjw5azABhD1ARIsAA0WFUEEG-O19kXLlC5LPJF0j3GZio8sp_XLW_QCTnrX72gvM3M-I1-CkYkaAn4WEALw_wcB:G:s&s_kwcid=AL!4422!3!639434067723!e!!g!!sagemaker!19155106685!149379722812&gbraid=0AAAAADjHtp-truM88nQvxnWFP4QDEYGTo&gclid=Cj0KCQjw5azABhD1ARIsAA0WFUEEG-O19kXLlC5LPJF0j3GZio8sp_XLW_QCTnrX72gvM3M-I1-CkYkaAn4WEALw_wcB)<br>
[S3](https://aws.amazon.com/tw/pm/serv-s3/?trk=d171c0b1-a233-43fd-a766-4ffdfd6f6398&sc_channel=ps&ef_id=Cj0KCQjw5azABhD1ARIsAA0WFUED7GcKQ9JFMAlm1mJcYlpzkHsPvFUkVTcDlE3k3ctdUvbX-RFnPYcaArLqEALw_wcB:G:s&s_kwcid=AL!4422!3!595905315986!e!!g!!s3!17115100962!136234441636&gbraid=0AAAAADjHtp-GBZEzo9SJ_FE9SCCBYzC2r&gclid=Cj0KCQjw5azABhD1ARIsAA0WFUED7GcKQ9JFMAlm1mJcYlpzkHsPvFUkVTcDlE3k3ctdUvbX-RFnPYcaArLqEALw_wcB)<br>
[Guardrails](https://docs.aws.amazon.com/zh_tw/bedrock/latest/userguide/guardrails-how.html)<br>
[Agent](https://aws.amazon.com/tw/bedrock/agents/)<br>
[Tavily](https://tavily.com/)<br>
[HuggingFace openai/gpt2](https://huggingface.co/openai-community/gpt2)<br>
[HuggingFace openai/whisper](https://huggingface.co/openai/whisper-large)<br>
[Rag](https://cloud.google.com/use-cases/retrieval-augmented-generation?hl=zh-TW)<br>