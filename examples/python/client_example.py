"""MembrainClient -- store and retrieve LLM memories with case-based reasoning."""

from membrain import MembrainClient


def main() -> None:
    with MembrainClient() as client:
        client.store_fact("Python was created by Guido van Rossum in 1991", confidence=0.95)
        client.store_preference(
            holder="user", subject="languages",
            preference="prefers strongly-typed languages", strength="strong",
        )
        client.store_entity(name="membrain", entity_type="software_project")
        client.store_concept(
            name="Vector Similarity Search",
            definition="Finding closest vectors in high-dimensional space",
        )

        client.store_case(
            problem="User asked to summarize a 10-page document",
            plan="Split into sections, summarize each, then combine",
            outcome="User satisfied with hierarchical summary",
            reward=1.0,
        )

        print(f"Stored {client.count()} memories")

        results = client.search("programming languages", limit=5)
        print(f"\nSearch results ({results.duration_ms}ms):")
        for memory in results.memories:
            print(f"  [{memory.memory_type}] {memory.content[:80]} (score={memory.score:.3f})")

        filtered = client.search("vector search", limit=5, filters={"memory_types": ["fact", "concept"]})
        print(f"\nFiltered (facts+concepts): {len(filtered.memories)} results")

        cases = client.search_cases("summarize a long document", limit=3, positive_reward_threshold=0.5)
        print(f"\nCases: {len(cases.positive_cases)} positive, {len(cases.negative_cases)} negative")


if __name__ == "__main__":
    main()
