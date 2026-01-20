import os
import json
import asyncio
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from mem0 import MemoryClient

# Optional DSPy-based memory extractor
try:
    import dspy
    from dspy import InputField, OutputField, Signature

    class MemoryExtract(Signature):
        """
        Extract relevant information from the conversation to store as long-term memory.
        """

        transcript: str = InputField()
        information: str = OutputField()

    memory_extractor = dspy.Predict(MemoryExtract)
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    memory_extractor = None


load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from .env
mem0_api_key = os.getenv("MEM0_API_KEY")  # Uses MEM0_API_KEY from .env
memory = MemoryClient(api_key=mem0_api_key)

user_id = "avb"


async def extract_memories_from_messages(messages: List[Dict[str, str]]) -> str:
    if not DSPY_AVAILABLE:
        return ""

    transcript = json.dumps(messages)

    # Use OpenAI via DSPy; switch model if you prefer another provider
    with dspy.context(lm=dspy.LM(model="openai/gpt-4o-mini")):
        result = await memory_extractor.acall(transcript=transcript)

    if result and getattr(result, "information", ""):
        return result.information
    return ""


async def main() -> None:
    messages: List[Dict[str, str]] = []

    while True:
        user_input = input("User: ")

        messages.append({"role": "user", "content": user_input})

        related_memories = memory.search(
            user_input,
            user_id=user_id,
            filters={"user_id": user_id},
        )
        print(related_memories)

        related_memories_text = (
            "\n".join([m["memory"] for m in related_memories["results"]])
            if related_memories["results"]
            else "No previous interactions found."
        )

        print(related_memories_text)

        system_message = [
            {
                "role": "system",
                "content": (
                    "answer the user's question honestly.\n"
                    "Here are some relevant information you may find useful that previous interactions have taught us:\n"
                    f"{related_memories_text}"
                ),
            }
        ]

        response = client.chat.completions.create(
            messages=system_message + messages,
            model="gpt-4o",
        )

        answer = response.choices[0].message.content

        messages.append({"role": "assistant", "content": answer})

        print(f"\nAssistant: {answer}\n")

        # Store raw turn
        memory.add(messages[-2:], user_id=user_id)

        # Store extracted long-term memory summary (if DSPy is available)
        try:
            summary = await extract_memories_from_messages(messages)
            if summary:
                memory.add(
                    [{"role": "system", "content": summary}],
                    user_id=user_id,
                )
        except Exception as err:  # Rarely hit; keep chat running
            print(f"Memory extraction warning: {err}")


if __name__ == "__main__":
    asyncio.run(main())


