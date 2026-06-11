"""Test direct Membrain usage without frameworks"""

from membrain import MembrainClient

def test_direct_usage():
    """Use Membrain directly without any framework"""
    client = MembrainClient()

    print("=" * 60)
    print("DIRECT USAGE TEST - No Framework Required")
    print("=" * 60)

    # Store different types of memories
    print("\n1. Storing memories...")
    result1 = client.store_fact("Python is a programming language", confidence=0.9)
    print(f"   Stored fact: {result1.id}")

    result2 = client.store_observation("User asked about LangChain integration")
    print(f"   Stored observation: {result2.id}")

    result3 = client.store_concept("RAG", "Retrieval-Augmented Generation")
    print(f"   Stored concept: {result3.id}")

    result4 = client.store_skill("code_review", "Analyze code quality")
    print(f"   Stored skill: {result4.id}")

    # Search memories
    print("\n2. Searching memories...")
    results = client.search("Python programming", limit=5)
    print(f"   Found {len(results.memories)} memories:")
    for memory in results.memories:
        print(f"   - {memory.memory_type}: {memory.content[:50]}... (score: {memory.score:.3f})")

    # Get stats
    print("\n3. Memory statistics:")
    print(f"   Total memories: {client.count()}")
    stats = client.stats()
    print(f"   Stats: {stats}")

    print("\n✓ Direct usage test completed successfully!\n")

if __name__ == "__main__":
    test_direct_usage()
