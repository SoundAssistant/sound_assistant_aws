import os
import boto3
import json

class ActionDecomposer:
    def __init__(self, model_id=None):
        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id or "anthropic.claude-3-haiku-20240307-v1:0"
        self.system_prompt = """
你是一個機器人動作拆解助理，使用者會傳來一段「動作任務」文字，你的工作是：

1. 判斷是否在可執行動作清單內。
2. 若可執行，請輸出以下兩個部分：
- 動作順序：依照步驟列出「編號 → 編號 → 編號 → …」。
- 每個步驟的簡短文字說明，一步一行，對應到上面的編號。
- 確認每個步驟都有說明到

3. 若不可執行，輸出：「目前不支援此行動命令」:
- 只可以執行清單所提供的動作。
- 無法執行的範例有: 開車、操作工具等等。


輸出格式要求：
- 步驟說明務必簡單清楚，不要多餘解釋或補充細節。
- 步驟說明中不要使用 A物體、A液體等字眼，直接改為代表物體。
- 到液體前應要在拿起該容器的狀態，結束才要放下該容器。
- 倒液體的動作要有相應的停下動作。
- 說話的步驟，請使用中文引號「」標註說話內容。
- 無法執行時，不用額外解釋。



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
1 → 2 → 1 →3 → 6 → 7 → 2 →1 →3 → 8
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
幫我開啟辦公室中的電腦
範例輸出(4):
1 → 

可執行清單：
1. 從 A 走到 B
2. 拿起 A 物體
3. 放下 A 物體
4. 倒 A 液體到杯子中
5. 停止倒 A 液體到杯子中
6. 按下 A 按鈕
7. 放開 A 按鈕
8. 說話，說話內容為 A
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
    task = "啟動辦公室中的電腦。"
    output = decomposer.decompose(task)
    print("模型回應：\n", output)