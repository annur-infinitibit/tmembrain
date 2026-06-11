"""MembrainGraph -- knowledge graph with multi-hop traversal and persistence."""

import random

from membrain import MembrainClient, MembrainGraph


def embed(seed: float, dimension: int = 128) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(dimension)]


def main() -> None:
    with MembrainClient() as client, MembrainGraph() as graph:
        topics = [
            ("Python is a high-level programming language", 0.95),
            ("Rust provides memory safety without garbage collection", 0.95),
            ("HNSW is a graph-based ANN algorithm", 0.90),
            ("Vector databases store high-dimensional embeddings", 0.90),
            ("RAG combines retrieval with generation", 0.85),
        ]

        for i, (statement, confidence) in enumerate(topics):
            result = client.store_fact(statement, confidence=confidence)
            if result.success and result.id:
                graph.add_node(result.id, embed(seed=float(i)), confidence=confidence)

        print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

        query = embed(seed=2.5)
        result = graph.query(query, max_hops=3, top_k=3)
        print(f"\nMulti-hop query: {result.hops_performed} hops, {result.nodes_visited} visited")
        for node in result.nodes:
            memory = client.get(node.memory_id)
            print(f"  {memory.content[:60]}... (score={node.score:.3f}, hop={node.hop_distance})")

        serialized = graph.save()
        print(f"\nSerialized: {len(serialized)} bytes")

    restored = MembrainGraph.load(serialized)
    print(f"Restored: {restored.node_count()} nodes, {restored.edge_count()} edges")
    restored.close()


if __name__ == "__main__":
    main()
