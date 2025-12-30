from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory
import os
import json

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {"api_key": OPEN_AI_API_KEY, "model": "text-embedding-3-large"},
    },
    "llm": {
        "provider": "openai",
        "config": {"api_key": OPEN_AI_API_KEY, "model": "gpt-4.1"},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("NEO4J_URI"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD"),
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
}
memory_client = Memory.from_config(config)

while True:

    user_query = input(">")
    search_memory = memory_client.search(user_id="Rajib", query=user_query)

    memory = [
        f"ID:{mem.get("id")}\nMemory:{mem.get("memory")}"
        for mem in search_memory.get("results", [])
    ]
    print("Relevant Memory:", memory)

    SYSTEM_MESSAGE = f"""
     Hey, use this memory to answer the user query.
     {json.dumps(memory)}
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_query},
        ],
    )
    ai_response = response.choices[0].message.content
    memory_client.add(
        user_id="Rajib",
        messages=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response},
        ],
    )
    print("Response:", ai_response)
