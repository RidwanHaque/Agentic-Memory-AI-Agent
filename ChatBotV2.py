import os
import json
import asyncio
from typing import Dict, List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from mem0 import MemoryClient
from pydantic import BaseModel

from databaes import generate_embeddings

# Optional DSPy-based memory extractor
try:
    import dspy
    from dspy import InputField, OutputField, Signature

    class Memory(BaseModel):
        information: str
        predicted_categories: List[str]
        sentiment: Literal["happy", "sad", "neutral"]

    class MemoryExtract(Signature):
        """
        Extract relevant information from the conversation. Create memory entries
        that you should remember when speaking with the user later.

        You will be provided a list of the existing categories in the memory
        database. When predicting the category of this information, you can decide
        to create new categories, or pick from an existing one if it exists.

        Extract information piece by piece creating atomic units of memory.

        If transcript does not contain any information worth extracting, set
        no_info to True, else False.
        """

        transcript: str = InputField()
        existing_categories: List[str] = InputField()
        no_info: bool = OutputField(description="set true if no info to be extracted")
        memories: List[Memory] = OutputField()

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
memory_categories: List[str] = []


async def extract_memories_from_messages(messages: List[Dict[str, str]], categories: List[str] = []) -> List[Memory]:
    if not DSPY_AVAILABLE:
        return []

    transcript = json.dumps(messages)

    # Use OpenAI via DSPy; switch model if you prefer another provider
    with dspy.context(lm=dspy.LM(model="openai/gpt-4o-mini")):
        result = await memory_extractor.acall(
            transcript=transcript,
            existing_categories=categories,
        )

    if result and not result.no_info and result.memories:
        return result.memories
    return []


async def embed_memories(memories: List[Memory]):
    memory_texts = [
        m.information
        for m in memories
    ]
    embeddings = await generate_embeddings(memory_texts)
    return embeddings


async def extract_and_embed_memories(
    messages,
    existing_categories
):
    memories = await extract_memories_from_messages(messages,
                                                      existing_categories)
    embeddings = await embed_memories(memories)
    return memories, embeddings


async def main() -> None:
    global memory_categories
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
            extracted_memories = await extract_memories_from_messages(messages, memory_categories)
            if extracted_memories:
                for mem in extracted_memories:
                    # Update categories
                    for cat in mem.predicted_categories:
                        if cat not in memory_categories:
                            memory_categories.append(cat)
                    
                    # Store structured memory
                    memory_content = f"{mem.information} [Sentiment: {mem.sentiment}, Categories: {', '.join(mem.predicted_categories)}]"
                    memory.add(
                        [{"role": "system", "content": memory_content}],
                        user_id=user_id,
                    )
                print(f"Extracted {len(extracted_memories)} memory(ies). Categories: {memory_categories}")
        except Exception as err:  # Rarely hit; keep chat running
            print(f"Memory extraction warning: {err}")


if __name__ == "__main__":
    asyncio.run(main())


