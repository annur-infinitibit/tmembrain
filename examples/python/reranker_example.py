"""Reranker -- re-score search results using LLM-based relevance.

Requires: OPENAI_API_KEY environment variable (or swap for another provider).
"""

import os

from membrain import MembrainClient, OpenAIReranker


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example.")
        return

    with MembrainClient() as client:
        for statement, confidence in [
            ("HNSW creates a multi-layered graph for ANN search", 0.95),
            ("Cosine similarity measures the angle between two vectors", 0.90),
            ("IVF partitions the vector space using k-means clustering", 0.85),
            ("LSH uses random projections for fast approximate search", 0.85),
            ("Product quantization compresses vectors by splitting into subspaces", 0.80),
        ]:
            client.store_fact(statement, confidence=confidence)

        query = "How does vector search work?"
        results = client.search(query, limit=5)

        print(f"Initial search for '{query}':")
        for memory in results.memories:
            print(f"  [{memory.score:.3f}] {memory.content[:70]}")

        reranker = OpenAIReranker(api_key=api_key, top_k=5)
        reranked = reranker.rerank(query, results, top_k=3)

        print(f"\nReranked (model={reranked.model}, {reranked.duration_ms}ms):")
        for memory in reranked.memories:
            print(f"  [{memory.relevance_score:.3f}] {memory.content[:70]}")


if __name__ == "__main__":
    main()
