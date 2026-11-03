from langchain_groq import ChatGroq
from typing import List,Dict
import os
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class ChatBot:

    def __init__(self,
                 api_key: str,
                 model_name: str = "openai/gpt-oss-20b",
                 temp: float = 0.1,
                 max_tokens: int = 2048,
                 max_history_turns: int = 10):
        print("Running ChatBot for you!!!")

        if not api_key:
            raise ValueError("Set GROQ_API_KEY in your environment or pass api_key.")

        self.api_key = api_key
        self.model_name = model_name
        self.temp = temp
        self.max_tokens = max_tokens
        self.max_history_turns = max_history_turns
        self.history: List[BaseMessage] = []
        self.system_prompt = """You are a helpful assistant. Give concise, direct answers.
         "If the question is ambiguous, ask a short clarifying question first."
         """

        self.llm = ChatGroq(api_key=self.api_key, model=self.model_name, temperature=self.temp, max_tokens=self.max_tokens)

    def _recent_history(self) -> List[BaseMessage]:
        # each turn = 2 messages (human + AI), so keep the last N turns
        return self.history[-2 * self.max_history_turns:]

    def ask(self, query: str) -> str:
        messages = [
            SystemMessage(self.system_prompt),
            *self._recent_history(),
            HumanMessage(query),
        ]
        result = self.llm.invoke(messages)   # result is the full AIMessage

        # Save the real response so metadata (and tool_calls later) is kept
        self.history.append(HumanMessage(query))
        self.history.append(result)

        if result.response_metadata.get("finish_reason") == "length":
            print("Warning: answer was cut off. Increase max_tokens.")
        print("token usage:", result.usage_metadata)

        return result.content                # callers just get the text

    def reset(self) -> None:
        self.history.clear()


if __name__ == "__main__":
    groq_api_key=os.getenv("GROQ_API_KEY")
    agent = ChatBot(api_key=groq_api_key)

    while True:
        query = input("ask question from bot (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        print(agent.ask(query))
        