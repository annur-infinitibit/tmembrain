"""Simple LangChain + Membrain integration test without complex abstractions"""

import os
from dotenv import load_dotenv
from membrain import MembrainClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def test_langchain_simple():
    print("=" * 60)
    print("LANGCHAIN + MEMBRAIN SIMPLE INTEGRATION TEST")
    print("=" * 60)

    # Initialize Membrain and LangChain LLM
    memory = MembrainClient()
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

    print("\n1. First interaction:")
    user_msg1 = "Hi! My name is Alice and I love Python programming."

    # Store in Membrain
    memory.store_observation(f"User: {user_msg1}")

    # Get LLM response
    messages = [
        SystemMessage(content="You are a helpful AI assistant with memory."),
        HumanMessage(content=user_msg1)
    ]
    response1 = llm.invoke(messages)
    memory.store_observation(f"AI: {response1.content}")

    print(f"   User: {user_msg1}")
    print(f"   AI: {response1.content[:100]}...")

    print("\n2. Second interaction with memory recall:")
    user_msg2 = "What's my name and what do I like?"

    # Search for relevant context from Membrain
    context_results = memory.search(user_msg2, limit=3)
    context = "\n".join([m.content for m in context_results.memories])

    # Build messages with context
    messages = [
        SystemMessage(content=f"You are a helpful AI assistant. Here's what you remember:\n{context}"),
        HumanMessage(content=user_msg2)
    ]
    response2 = llm.invoke(messages)
    memory.store_observation(f"User: {user_msg2}")
    memory.store_observation(f"AI: {response2.content}")

    print(f"   User: {user_msg2}")
    print(f"   AI: {response2.content}")

    print("\n3. Memory statistics:")
    stats = memory.stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Memory types: {list(stats['by_type'].keys())}")

    print("\n✓ LangChain simple integration test completed!")

if __name__ == "__main__":
    test_langchain_simple()
