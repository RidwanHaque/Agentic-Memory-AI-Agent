from datetime import datetime
import openai
import json
import asyncio

from databaes import generate_embeddings
from vectordb import (
    EmbeddedMemory,
    delete_records,
    insert_memories,
    search_memories,
    stringify_retrieved_point,
    create_memory_collection,
    get_all_categories,
)
from ChatBotV2 import extract_and_embed_memories


client = openai.AsyncOpenAI()
user_id = 1


async def fill_up_context(messages):
    """Extract and embed memories from messages, then store them in vector DB."""
    infos, embeds = await extract_and_embed_memories(messages, [])
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    memories = [
        EmbeddedMemory(
            user_id=user_id,
            memory_text=info.information,
            categories=info.predicted_categories,
            date=current_date,
            embedding=embed,
        )
        for info, embed in zip(infos, embeds)
    ]
    print("\n".join([f"{m.memory_text} ({m.categories})" for m in memories]))
    await insert_memories(memories)


async def run_test_question(question):
    """Run a test question against stored memories."""
    # Generate embedding for the question
    question_vector = (await generate_embeddings([question]))[0]
    
    # Search for relevant memories
    retrieved_memories = await search_memories(
        search_vector=question_vector,
        user_id=user_id,
        categories=None,
    )
    
    # Format retrieved memories for context
    context = "\n".join(
        [stringify_retrieved_point(mem) for mem in retrieved_memories]
    )
    
    # Build messages for LLM
    system_message = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant. Use the following memories if relevant:
{context if context else "No relevant memories found."}

Answer the user's question based on the memories above and general knowledge.""",
        }
    ]
    
    messages = system_message + [{"role": "user", "content": question}]
    
    # Get response from GPT
    response = await client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )
    
    answer = response.choices[0].message.content
    return answer


async def run_evaluation(test_file: str):
    """Load test file and run evaluation."""
    # Initialize collection
    await create_memory_collection()
    
    # Load test data
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    context_messages = test_data.get("context", [])
    tests = test_data.get("test", [])
    
    # Fill up context with memories
    print("=" * 50)
    print("STORING MEMORIES FROM CONTEXT")
    print("=" * 50)
    await fill_up_context(context_messages)
    
    # Run tests
    print("\n" + "=" * 50)
    print("RUNNING TESTS")
    print("=" * 50)
    
    results = []
    for i, test in enumerate(tests, 1):
        question = test.get("question", "")
        expected_answer = test.get("answer", "")
        
        print(f"\nTest {i}:")
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        
        # Run the test
        actual_answer = await run_test_question(question)
        print(f"Actual: {actual_answer}")
        
        # Check if answer contains key information
        match = expected_answer.lower() in actual_answer.lower()
        results.append({
            "test_num": i,
            "question": question,
            "expected": expected_answer,
            "actual": actual_answer,
            "passed": match,
        })
        
        print(f"Passed: {'✓' if match else '✗'}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    # Optional: Clean up user records (commented out to persist memories for dashboard inspection)
    # print("\nCleaning up test data...")
    # from vectordb import fetch_all_user_records
    # all_records = await fetch_all_user_records(user_id)
    # if all_records:
    #     point_ids = [rec.point_id for rec in all_records]
    #     await delete_records(point_ids)
    
    return results


if __name__ == "__main__":
    # Run with test1.json
    results = asyncio.run(run_evaluation("test1.json"))
