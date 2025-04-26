import os
import boto3
import json

class ActionDecomposer:
    def __init__(self, model_id=None):
        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id or "anthropic.claude-3-haiku-20240307-v1:0"
        self.system_prompt = """
你是一個機器人動作拆解助理。使用者會傳來一段「動作任務」文字，你的工作是：
1. 判斷是否在可執行動作清單內。
2. 若可執行，就輸出「動作順序：編號 → 編號 → …」以及對應的每步文字說明。
3. 若不可執行，輸出「目前不支援此行動命令」。

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
    task = "幫我把這杯水倒入杯子中，然後跟使用者說「完成了」。"
    output = decomposer.decompose(task)
    print("模型回應：\n", output)