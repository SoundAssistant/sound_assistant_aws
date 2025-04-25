import json
import time
from typing import List, Dict, Optional
from botocore.exceptions import ClientError

# --- RAG SYSTEM CLASSES ---

class Retriever:
    def __init__(self, agent_client, knowledge_base_id: str, number_of_results: int = 5):
        self.agent_client = agent_client
        self.knowledge_base_id = knowledge_base_id
        self.number_of_results = number_of_results

    def retrieve(self, query: str) -> List[str]:
        results = self.agent_client.retrieve(
            retrievalQuery={'text': query},
            knowledgeBaseId=self.knowledge_base_id,
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': self.number_of_results
                }
            }
        )
        return [result['content']['text'] for result in results['retrievalResults']]


class PromptBuilder:
    @staticmethod
    def build_prompt(contexts: List[str], query: str) -> str:
        return f"""
            Here is the context to reference:
            <context>
            {chr(10).join(contexts)}
            </context>

            Referencing the context, answer the user question
            <question>
            {query}
            </question>
            """


class ConversationalModel:
    def __init__(self, client, model_id: str, temperature: float = 0.5, top_k: int = 200):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.top_k = top_k
        self.system_prompts = [{"text": "You are a Question and answering assistant and your responsibility is to answer user questions based on provided context."}]

    def converse(self, messages: List[Dict]) -> Dict:
        response = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=self.system_prompts,
            inferenceConfig={"temperature": self.temperature},
            additionalModelRequestFields={"top_k": self.top_k}
        )
        return response['output']['message']


class RAGPipeline:
    def __init__(self, retriever: Retriever, model: ConversationalModel):
        self.retriever = retriever
        self.model = model
        self.messages = []

    def answer(self, query: str) -> str:
        contexts = self.retriever.retrieve(query)
        prompt = PromptBuilder.build_prompt(contexts, query)
        user_msg = {"role": "user", "content": [{"text": prompt}]}
        self.messages.append(user_msg)

        max_retries = 5
        initial_delay = 1

        for attempt in range(max_retries):
            try:
                response = self.model.converse(self.messages)
                self.messages.append(response)
                return response['content'][0]['text']
            except ClientError as e:
                if attempt == max_retries - 1:
                    raise e
                delay = initial_delay * (2 ** attempt)
                print(f"Throttled. Retrying in {delay} seconds...")
                time.sleep(delay)


# --- SAMPLE USAGE ---

if __name__ == '__main__':
    import streamlit as st
    from utils import BEDROCK_CLIENT, BEDROCK_AGENT_CLIENT, KNOWLEDGE_BASE_ID

    retriever = Retriever(BEDROCK_AGENT_CLIENT, KNOWLEDGE_BASE_ID)
    model = ConversationalModel(BEDROCK_CLIENT, model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    rag_pipeline = RAGPipeline(retriever, model)

    st.title("RAG-powered Q&A")
    user_query = st.text_input("Ask a question:")

    if user_query:
        answer = rag_pipeline.answer(user_query)
        st.write("Answer:", answer)
