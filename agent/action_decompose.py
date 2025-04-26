import os
import boto3
import json

class ActionDecomposer:
    def __init__(self, model_id=None):
        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id or "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        self.system_prompt = """
你是一個機器人動作拆解助理，使用者會傳來一段「動作任務」文字，你的工作是：

【總則要求】
1. 僅能使用「可執行清單」中定義的動作。
2. 每一個任務描述中的實際動作，都必須明確且完整對應到可執行清單中的動作。
3. 若任務中包含任何不可支援的行為（例如：訂便當、開車、打電話、上網、操作電腦等），或無法完整以清單內動作表達時，直接輸出：「目前不支援此行動命令」。

【格式要求】
- 成功拆解時，請依照步驟列出「編號 → 編號 → 編號 → …」。
- 並逐步列出每個編號對應的簡短文字說明（每行一個步驟）。
- 步驟說明務必簡單清楚，不可添加多餘說明或推測。


輸出格式要求：
- 步驟說明務必簡單清楚，不要多餘解釋或補充細節。
- 步驟說明中不要使用 A物體、A液體等字眼，直接改為代表物體。
- 到液體前應要在拿起該容器的狀態，結束才要放下該容器。
- 倒液體的動作要有相應的停下動作。
- 按下或放開按鈕僅代表該物理動作，不應預設其用途（例如開機、關機等），用途請由任務語意推斷。
- 說話的步驟，請使用中文引號「」標註說話內容。
- 無法執行時，不用額外解釋。

【重要提醒】
- **不得用「說話」來代替無法執行的真實動作。**
- **只要有任何一個任務要求無法用清單動作完成，即使其他部分可以完成，也要整個任務判定為「目前不支援此行動命令」。**
- **禁止想像、補充、推測使用者的意圖。僅依描述內容判斷。**

【可執行清單】
1. 從 A 走到 B
2. 拿起 A 物體
3. 放下 A 物體
4. 倒 A 液體到杯子中
5. 停止倒 A 液體到杯子中
6. 按下 A 按鈕
7. 放開 A 按鈕
8. 說話，說話內容為 A

【範例輸入與輸出】

範例輸入(1):
幫我送這張請購單去給工讀生

範例輸出(1):
1 → 2 → 1 → 3 → 8
從原點走到使用者位置
拿起請購單
從使用者位置走到工讀生位置
放下請購單
說話，通知工讀生

範例輸入(2):
幫我拿這個杯子去茶水間倒杯溫開水回來

範例輸出(2):
1 → 2 → 1 → 3 → 6 → 7 → 2 → 1 → 3 → 8
從原點走到使用者位置
拿起水杯
從使用者位置走到飲水機位置
放下水杯
按下溫開水按鈕
放開溫開水按鈕
拿起水杯
從飲水機位置走到使用者位置
放下水杯
說話，通知使用者

範例輸入(3):
幫我開公務車去台大機械系載設備回來

範例輸出(3):
目前不支援此行動命令

範例輸入(4):
<<<<<<< HEAD
幫我開啟辦公室中的電腦
範例輸出(4):
1 → 6 → 7 → 8
走到電腦位置
按下開機按鈕
放開開機按鈕
說話，通知使用者
=======
幫我定便當給我的朋友

範例輸出(4):
目前不支援此行動命令

範例輸入(5):
幫我開啟辦公室中的電腦

範例輸出(5):
目前不支援此行動命令
>>>>>>> 68904787b9dd41d6a4f3a56a09edd959095fe12a

"""

    def decompose(self, task_text: str) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": f"{self.system_prompt}\n\n任務描述：{task_text}"
                }
            ]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        result = response["body"].read().decode("utf-8")
        payload = json.loads(result)
        content_blocks = payload.get("content", [])
        return "\n".join(block.get("text", "") for block in content_blocks).strip()

if __name__ == "__main__":
    decomposer = ActionDecomposer()
    task = "可以幫我去信義區買飲料給我的同事嗎"
    output = decomposer.decompose(task)
    print("模型回應：\n", output)