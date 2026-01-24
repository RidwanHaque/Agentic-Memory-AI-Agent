from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, models
import asyncio

client = AsyncQdrantClient(url="http://localhost:6333")


async def create_memory_collection():
    if not (await client.collection_exists("memories")):
        await client.create_collection(
            collection_name="memories",
            vectors_config=VectorParams(size=64, distance=Distance.DOT),
        )

        await client.create_payload_index(
            collection_name="memories",
            field_name="user_id",
            field_schema=models.PayloadSchemaType.INTEGER
        )

        # Create an index on the 'categories' field
        await client.create_payload_index(
            collection_name="memories",
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        print("Collection created")
    else:
        print("Collection exists")


if __name__ == "__main__":
    asyncio.run(create_memory_collection())
