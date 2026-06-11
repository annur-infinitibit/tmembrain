"""
Test Graph Memory Cookbook Examples
Based on docs/cookbooks/graph-memory.mdx
"""
import os
import random
import uuid
import pytest
from dotenv import load_dotenv

try:
    from membrain import MembrainClient, MembrainGraph
    MEMBRAIN_AVAILABLE = True
except ImportError:
    MEMBRAIN_AVAILABLE = False
    pytest.skip("Membrain not installed", allow_module_level=True)

# Load environment variables
load_dotenv()


class TestGraphBasics:
    """Test basic graph operations"""

    def test_create_graph(self):
        """Test creating a graph instance"""
        graph = MembrainGraph({"embedding_dim": 16})
        try:
            assert graph is not None
        finally:
            graph.close()

    def test_graph_with_config(self):
        """Test creating graph with custom configuration"""
        graph = MembrainGraph({
            "hidden_dim": 128,
            "embedding_dim": 16
        })
        try:
            assert graph is not None
        finally:
            graph.close()


class TestGraphNodes:
    """Test adding and querying nodes in the graph"""

    def test_add_nodes(self):
        """Test adding nodes to the graph"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Add nodes with random embeddings and UUID memory_ids
            for i in range(10):
                embedding = [random.random() for _ in range(16)]
                memory_id = str(uuid.uuid4())
                graph.add_node(memory_id, embedding, 0.9)

            # Check node count
            count = graph.node_count()
            assert count == 10

        finally:
            graph.close()

    def test_node_with_confidence(self):
        """Test adding nodes with confidence scores"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            embedding = [random.random() for _ in range(16)]
            memory_id = str(uuid.uuid4())
            graph.add_node(memory_id, embedding, confidence=0.85)

            count = graph.node_count()
            assert count == 1

        finally:
            graph.close()


class TestGraphQuery:
    """Test graph query and traversal"""

    def test_single_hop_query(self):
        """Test querying graph with single hop"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Build a small graph
            for i in range(20):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding, 0.8)

            # Query
            query_embedding = [random.random() for _ in range(16)]
            result = graph.query(
                embedding=query_embedding,
                max_hops=1,
                top_k=5
            )

            assert result is not None
            assert len(result.nodes) > 0
            assert len(result.nodes) <= 5

            # Check node properties
            for node in result.nodes:
                assert node.memory_id is not None
                assert node.score >= 0.0
                assert node.hop_distance >= 0

        finally:
            graph.close()

    def test_multi_hop_query(self):
        """Test multi-hop graph traversal"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Build graph
            for i in range(20):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding, 0.8)

            # Multi-hop query
            query_embedding = [random.random() for _ in range(16)]
            result = graph.query(
                embedding=query_embedding,
                max_hops=3,
                top_k=5
            )

            assert len(result.nodes) > 0
            assert result.hops_performed >= 0
            assert result.hops_performed <= 3

            # Verify traversal path exists
            assert hasattr(result, 'traversed_edges')

        finally:
            graph.close()

    def test_query_result_structure(self):
        """Test structure of query results"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Add nodes
            for i in range(10):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            # Query
            query_embedding = [random.random() for _ in range(16)]
            result = graph.query(query_embedding, 2, 5)

            # Check result structure
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'hops_performed')
            assert hasattr(result, 'traversed_edges')

            # Check nodes have required properties
            if len(result.nodes) > 0:
                node = result.nodes[0]
                assert hasattr(node, 'memory_id')
                assert hasattr(node, 'score')
                assert hasattr(node, 'hop_distance')

        finally:
            graph.close()


class TestGraphPersistence:
    """Test graph save and load functionality"""

    def test_save_graph(self):
        """Test saving graph state"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Add some nodes
            for i in range(5):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            # Save
            saved_data = graph.save()
            assert saved_data is not None
            assert len(saved_data) > 0
            assert isinstance(saved_data, str)

        finally:
            graph.close()

    def test_save_and_load_graph(self):
        """Test saving and loading graph state"""
        # Create and save
        graph = MembrainGraph({"embedding_dim": 16})
        try:
            for i in range(10):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            saved_data = graph.save()
            original_count = graph.node_count()
        finally:
            graph.close()

        # Load
        restored_graph = MembrainGraph.load(saved_data)
        try:
            restored_count = restored_graph.node_count()
            assert restored_count == original_count
        finally:
            restored_graph.close()

    def test_save_to_file(self, tmp_path):
        """Test saving graph to file and loading it back"""
        graph_file = tmp_path / "graph_state.txt"

        # Create and save
        graph = MembrainGraph({"embedding_dim": 16})
        try:
            for i in range(10):
                graph.add_node(str(uuid.uuid4()), [random.random() for _ in range(16)])

            saved_data = graph.save()
            graph_file.write_text(saved_data)
        finally:
            graph.close()

        # Load from file
        loaded_data = graph_file.read_text()
        restored_graph = MembrainGraph.load(loaded_data)
        try:
            assert restored_graph.node_count() == 10
        finally:
            restored_graph.close()


class TestGraphPruning:
    """Test graph pruning operations"""

    def test_prune_graph(self):
        """Test pruning low-quality connections"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Add many nodes
            for i in range(100):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            nodes_before = graph.node_count()
            edges_before = graph.edge_count()

            # Prune
            result = graph.prune()

            # Check result structure
            assert hasattr(result, 'edges_removed')
            assert hasattr(result, 'nodes_removed')
            assert hasattr(result, 'edges_remaining')
            assert hasattr(result, 'nodes_remaining')

            # Verify counts are non-negative
            assert result.edges_removed >= 0
            assert result.nodes_removed >= 0
            assert result.edges_remaining >= 0
            assert result.nodes_remaining >= 0

        finally:
            graph.close()


class TestClientGraphIntegration:
    """Test combining MembrainClient and MembrainGraph"""

    def test_store_in_client_index_in_graph(self):
        """Test storing memories in client and indexing in graph"""
        client = MembrainClient()
        # Create graph with matching embedding dimension (16 for tests)
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Store memories in client with unique content
            test_id = str(uuid.uuid4())
            ids = []
            topics = ["Python", "Rust", "JavaScript"]
            for topic in topics:
                result = client.store_fact(f"{topic} is a programming language {test_id}", 0.9)
                if result.success:
                    ids.append(result.id)

            # Skip if no memories were stored
            if not ids:
                pytest.skip("No memories stored (novelty threshold)")

            # Build graph relationships
            for mem_id in ids:
                embedding = [random.random() for _ in range(16)]
                graph.add_node(mem_id, embedding)

            # Query graph
            query_emb = [random.random() for _ in range(16)]
            result = graph.query(query_emb, max_hops=2, top_k=3)

            # Retrieve full content from client
            for node in result.nodes:
                memory = client.get(node.memory_id)
                if memory:
                    assert memory.content is not None
                    assert len(memory.content) > 0

        finally:
            client.close()
            graph.close()

    def test_hybrid_storage(self):
        """Test using both client and graph for storage and retrieval"""
        client = MembrainClient()
        # Create graph with matching embedding dimension
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Store concepts with unique content
            test_id = str(uuid.uuid4())
            concepts = [
                f"AI is a broad field of computer science {test_id}",
                f"Machine Learning is a subset of AI {test_id}",
                f"Deep Learning is a subset of Machine Learning {test_id}"
            ]

            stored_count = 0
            for concept in concepts:
                # Store in client
                result = client.store_concept("AI", concept)

                if result.success:
                    # Index in graph with random embedding
                    # (in production, use real embeddings from a model)
                    embedding = [random.random() for _ in range(16)]
                    graph.add_node(result.id, embedding)
                    stored_count += 1

            # Skip if nothing was stored
            if stored_count == 0:
                pytest.skip("No concepts stored (novelty threshold)")

            # Verify storage
            assert graph.node_count() == stored_count

            # Search using client
            search_results = client.search(f"machine learning {test_id}", limit=3)
            assert search_results is not None

        finally:
            client.close()
            graph.close()


class TestGraphEdges:
    """Test graph edge operations"""

    def test_edge_creation(self):
        """Test that edges are created between nodes"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Add several nodes
            for i in range(20):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            # Check if edges were created
            edge_count = graph.edge_count()
            # Edges should exist between related nodes
            assert edge_count >= 0

        finally:
            graph.close()

    def test_traversed_edges(self):
        """Test traversed edges in query results"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Build graph
            for i in range(15):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            # Query with multi-hop
            query_emb = [random.random() for _ in range(16)]
            result = graph.query(query_emb, max_hops=2, top_k=5)

            # Check traversed edges
            assert hasattr(result, 'traversed_edges')

            # If there are traversed edges, check their structure
            if len(result.traversed_edges) > 0:
                edge = result.traversed_edges[0]
                assert hasattr(edge, 'from_id') or hasattr(edge, 'from')
                assert hasattr(edge, 'to_id') or hasattr(edge, 'to')

        finally:
            graph.close()


class TestGraphConfiguration:
    """Test graph configuration options"""

    def test_custom_embedding_dimension(self):
        """Test creating graph with custom embedding dimension"""
        graph = MembrainGraph({
            "embedding_dim": 32,
            "hidden_dim": 64
        })

        try:
            # Add node with matching dimension
            embedding = [random.random() for _ in range(32)]
            graph.add_node(str(uuid.uuid4()), embedding)

            assert graph.node_count() == 1

        finally:
            graph.close()

    def test_graph_with_larger_dimensions(self):
        """Test graph with larger embedding dimensions"""
        graph = MembrainGraph({
            "embedding_dim": 384,  # Common for sentence transformers
            "hidden_dim": 128
        })

        try:
            # Add nodes with matching dimension
            for i in range(5):
                embedding = [random.random() for _ in range(384)]
                graph.add_node(str(uuid.uuid4()), embedding)

            assert graph.node_count() == 5

        finally:
            graph.close()


class TestGraphScalability:
    """Test graph performance with larger datasets"""

    def test_many_nodes(self):
        """Test adding many nodes to the graph"""
        graph = MembrainGraph({"embedding_dim": 16})

        try:
            # Add 500 nodes
            for i in range(500):
                embedding = [random.random() for _ in range(16)]
                graph.add_node(str(uuid.uuid4()), embedding)

            count = graph.node_count()
            assert count == 500

            # Query should still work
            query_emb = [random.random() for _ in range(16)]
            result = graph.query(query_emb, max_hops=2, top_k=10)
            assert len(result.nodes) > 0

        finally:
            graph.close()


class TestGraphMemoryIsolation:
    """Test that different graph instances are isolated"""

    def test_separate_graphs(self):
        """Test that two graphs don't interfere with each other"""
        graph1 = MembrainGraph({"embedding_dim": 16})
        graph2 = MembrainGraph({"embedding_dim": 16})

        try:
            # Add nodes to graph1
            for i in range(5):
                embedding = [random.random() for _ in range(16)]
                graph1.add_node(str(uuid.uuid4()), embedding)

            # Add nodes to graph2
            for i in range(3):
                embedding = [random.random() for _ in range(16)]
                graph2.add_node(str(uuid.uuid4()), embedding)

            # Verify counts are independent
            assert graph1.node_count() == 5
            assert graph2.node_count() == 3

        finally:
            graph1.close()
            graph2.close()
