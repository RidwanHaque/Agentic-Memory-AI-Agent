import os
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import MemoryClient


load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from .env
mem0_api_key = os.getenv("MEM0_API_KEY") # Uses MEM0_API_KEY from .env
memory_client = MemoryClient(api_key=mem0_api_key)

messages = [
    {
        "role": "system",
        "content": "be truthful and engaging in your response",
    },
]

while True:
    user_input = input("User: ")
    
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    
    response = client.chat.completions.create(
        messages=messages,
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


