import asyncio
import openai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = openai.AsyncClient()

async def generate_embeddings(strings: list[str]):
    print(strings)
    out = await client.embeddings.create(
        input=strings,
        model="text-embedding-3-small",
        dimensions=64,
    )
    embeddings = [item.embedding for item in out.data]
    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


if __name__ == "__main__":
    texts = [
        "Hello how are you",
        "I like Machine Learning",
    ]
    asyncio.run(generate_embeddings(texts))
