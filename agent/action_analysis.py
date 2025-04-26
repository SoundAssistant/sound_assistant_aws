import os
import boto3
import json

client = boto3.client("bedrock-runtime")

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

SYSTEM_PROMPT = """
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

def decompose_action(task_text: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n任務描述：{task_text}"}
        ]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = response["body"].read().decode("utf-8")
    payload = json.loads(result)
    content_blocks = payload.get("content", [])
    text_response = "\n".join(block.get("text", "") for block in content_blocks)
    return text_response.strip()

if __name__ == "__main__":
    task = "幫我把這杯水倒入杯子中，然後跟使用者說「完成了」。"
    output = decompose_action(task)
    print("模型回應：\n", output)
    task = "NBA怎麼報名"
    output = decompose_action(task)
    print("模型回應：\n", output)
