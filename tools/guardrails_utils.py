import os
from dotenv import load_dotenv
from typing import List, Optional
from client_utils import get_bedrock_agent_runtime_client

load_dotenv()

class GuardrailChecker:
    def __init__(self):
        self.guardrail_identifier = os.getenv("guardrail_identifier")
        self.guardrail_version = os.getenv("guardrail_version")
        self.agent_client = get_bedrock_agent_runtime_client()

    def check_text(self, text: str) -> Optional[List[str]]:
        response = self.agent_client.assess(
            guardrailIdentifier=self.guardrail_identifier,
            guardrailVersion=self.guardrail_version,
            input={"text": text},
            trace="ENABLED"
        )

        assessments = response.get("guardrailEvaluation", {}).get("inputAssessment", {}).get("topicPolicy", {}).get("topics", [])
        
        if assessments:
            return [topic.get("name") for topic in assessments if "name" in topic]
        return None

if __name__ == "__main__":
    checker = GuardrailChecker()
    query = "How to access the dark web safely?"

    result = checker.check_text(query)
    if result:
        print(f"Guardrail blocked. Topics matched: {result}")
    else:
        print("No policy violation.")