import json
from typing import Tuple
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.client_utils import get_bedrock_runtime_client  # ✅ 用你的 function 取 client

class TaskClassifier:
    def __init__(self, model_id: str = 'anthropic.claude-3-haiku-20240307-v1:0'):
        self.model_id = model_id
        self.accept = 'application/json'
        self.content_type = 'application/json'
        self.TASK_CLASSIFICATION_PROMPT = """
            請根據使用者輸入的內容，判斷該屬於哪一種類型的任務。
                輸入：
                你會收到一段使用者輸入的文字，這段文字會屬於以下三種任務類型之一。  
                請判斷是哪一種類型，只需要回覆任務名稱。

                三種任務類型：

                1. 查詢(Query):
                如果使用者是在詢問資訊(具有明確答案)、尋求解答、了解事實或知識，請分類為「查詢」。
                範例：  
                - 「今天天氣怎麼樣？」  
                - 「請解釋一下機器學習是什麼。」  
                - 「2022年世界盃冠軍是誰?」
                - 「幫我查行天宮附近的披薩店」
                - 「請問今天天氣如何?」
                - 「今天有什麼重大新聞?」

                2. 聊天(Chat):
                如果使用者是在進行閒聊(無明確答案)、打招呼、開玩笑、分享心情，請分類為「聊天」。
                範例：  
                - 「嗨，你好嗎？」  
                - 「講個笑話來聽聽！」  
                - 「我今天心情有點不好。」
                - 「你覺得 NBA 的 goat 是誰?」
                - 「你覺得奶茶跟拿鐵哪個好喝?」
                - 「我今天心情不好不想上班」 

                3. 行動（Action）：
                如果使用者是在要求執行某個指令或具體行動，請分類為「行動」。
                範例：  
                - 「幫我發封信給我老闆。」  
                - 「預訂一張飛往東京的機票。」  
                - 「把客廳的燈關掉。」
                - 「幫我去樓下拿包裹」
                - 「幫我去茶水間泡杯咖啡給顧問」
                - 「幫我送請購單給工讀生」

                回覆要求：
                - 請務必回覆兩個部分，並使用以下格式包住：
                    <class>任務名稱（查詢 / 聊天 / 行動）</class>
                    <extra>簡短說明為何這樣分類</extra>

                - 除了這兩個標籤包起來的內容之外，不要輸出其他文字。
            """
        self.system_prompt = self.TASK_CLASSIFICATION_PROMPT
        self.client = get_bedrock_runtime_client() 

    def _parse_tag(self, text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_index = text.find(start_tag)
        end_index = text.find(end_tag)
        if start_index == -1 or end_index == -1:
            return ""
        return text[start_index + len(start_tag):end_index].strip()

    def classify_task(self, task_description: str) -> Tuple[str, str]:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.0,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": task_description}],
                }
            ]
        }

        response = self.client.invoke_model(
            body=json.dumps(payload),
            modelId=self.model_id,
            accept=self.accept,
            contentType=self.content_type
        )

        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]

        task_class = self._parse_tag(response_text, "class")
        extra_info = self._parse_tag(response_text, "extra")

        return task_class, extra_info

if __name__ == "__main__":
    classifier = TaskClassifier()
    task = "請給我今天的股票價格誰最高"
    task_class, extra_info = classifier.classify_task(task)
    print(f"Task: {task}\nTask Class: {task_class}\nExtra Info: {extra_info}")
