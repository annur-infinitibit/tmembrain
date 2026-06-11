"""
LangChain + Membrain Integration Example

This shows how to use Membrain as a memory backend for LangChain conversations.
Membrain provides persistent, searchable memory across sessions.
"""

from membrain import MembrainClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize Membrain client
memory = MembrainClient()

# Initialize LangChain LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")


def chat_with_memory(user_input: str) -> str:
    """Send a message and get response with memory context"""

    # Search Membrain for relevant context
    context_results = memory.search(user_input, limit=5)
    context = "\n".join([m.content for m in context_results.memories])

    # Build message with context from memory
    messages = [
        SystemMessage(content=f"You are a helpful assistant. Previous context:\n{context}"),
        HumanMessage(content=user_input)
    ]

    # Get LLM response
    response = llm.invoke(messages)

    # Store interaction in Membrain
    memory.store_observation(f"User: {user_input}")
    memory.store_observation(f"AI: {response.content}")

    return response.content


# Example usage
if __name__ == "__main__":
    print("LangChain + Membrain Chat\n")

    # First message
    response1 = chat_with_memory("My name is Alice and I love Python")
    print(f"User: My name is Alice and I love Python")
    print(f"AI: {response1}\n")

    # Second message - should remember
    response2 = chat_with_memory("What's my name and what language do I like?")
    print(f"User: What's my name and what language do I like?")
    print(f"AI: {response2}\n")

    # Check memory stats
    print(f"Total memories stored: {memory.count()}")
