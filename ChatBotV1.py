import os
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import MemoryClient


load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from .env
mem0_api_key = os.getenv("MEM0_API_KEY") # Uses MEM0_API_KEY from .env
memory = MemoryClient(api_key=mem0_api_key)

user_id = "avb"

messages = []

while True:
    user_input = input("User: ")
    
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    
    related_memories = memory.search(user_input, user_id=user_id, filters={"user_id": user_id})
    print(related_memories)
    
    related_memories_text = "\n".join([m['memory'] for m in related_memories['results']]) if related_memories['results'] else "No previous interactions found."
    
    print(related_memories_text)
    
    system_message = [
        {
            "role": "system",
            "content": f"""answer the user's question honestly. 
Here are some relevant information you may find useful that previous interactions have taught us: 
{related_memories_text}"""
        }
    ]
    
    response = client.chat.completions.create(
        messages=system_message + messages,
        model="gpt-4o",
    )
    
    answer = response.choices[0].message.content
    
    messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    
    print(f"\nAssistant: {answer}\n")
    
    memory.add(messages[-2:], user_id=user_id)
    
    memory.add(messages[-2:], user_id=user_id)


