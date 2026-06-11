"""
LlamaIndex + Membrain Integration Example

This shows how to use Membrain with LlamaIndex for RAG applications.
Membrain stores conversation history and facts.
"""

from membrain import MembrainClient
from llama_index.llms.openai import OpenAI

# Initialize Membrain and LlamaIndex
memory = MembrainClient()
llm = OpenAI(model="gpt-4o-mini")


def chat_with_rag(user_query: str) -> str:
    """Query with retrieval-augmented generation"""

    # Search Membrain for relevant facts and context
    results = memory.search(user_query, limit=5)
    context = "\n".join([f"- {m.content}" for m in results.memories])

    # Build prompt with retrieved context
    prompt = f"""Based on this context:
{context}

Answer the question: {user_query}"""

    # Get LLM response
    response = llm.complete(prompt)

    # Store the interaction
    memory.store_observation(f"Query: {user_query}")
    memory.store_observation(f"Answer: {response.text}")

    return response.text


# Example usage
if __name__ == "__main__":
    print("LlamaIndex + Membrain RAG\n")

    # Store some facts
    memory.store_fact("Python is a high-level programming language", confidence=0.9)
    memory.store_fact("LangChain is a framework for LLM applications", confidence=0.9)
    memory.store_fact("RAG stands for Retrieval-Augmented Generation", confidence=0.9)

    # Query with RAG
    answer = chat_with_rag("What is Python?")
    print(f"Q: What is Python?")
    print(f"A: {answer}\n")

    print(f"Total memories: {memory.count()}")
