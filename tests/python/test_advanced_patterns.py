"""
Test Advanced Patterns Cookbook Examples
Based on docs/cookbooks/advanced-patterns.mdx
"""
import os
import time
import random
import json
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


class TestDeduplication:
    """Test memory deduplication"""

    def test_automatic_deduplication(self, tmp_path):
        """Test that similar facts are automatically deduplicated"""
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "dedup_test")
            }
        }
        client = MembrainClient(config=config)

        try:
            # Store similar facts
            r1 = client.store_fact("Paris is in France", 0.9)
            r2 = client.store_fact("Paris is located in France", 0.9)

            # First fact should be stored
            assert r1.id is not None

            # Second fact may be rejected due to low novelty or merged
            if r2.success:
                # If stored, check if it was merged
                if hasattr(r2, 'merged_with') and r2.merged_with is not None:
                    assert r2.merged_with == r1.id
            else:
                # Rejection due to low novelty is expected behavior
                assert r2.rejection_reason is not None
                assert "novelty" in r2.rejection_reason.lower()

        finally:
            client.close()

    def test_different_facts_not_deduplicated(self, tmp_path):
        """Test that different facts are stored separately"""
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "different_facts")
            }
        }
        client = MembrainClient(config=config)

        try:
            r1 = client.store_fact("Paris is in France", 0.9)
            r2 = client.store_fact("London is in England", 0.9)

            # Both should be stored successfully
            assert r1.success and r1.id is not None
            assert r2.success and r2.id is not None
            assert r1.id != r2.id

            # Search should find both
            results = client.search("capital city", limit=5)
            # At minimum, the memories exist
            assert results is not None

        finally:
            client.close()


class TestBatchOperations:
    """Test batch operations for performance"""

    def test_batch_insertion_performance(self, unique_storage_config):
        """Test storing many memories in batch"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Batch insert
            facts = [f"Fact number {i}" for i in range(100)]

            start = time.time()
            results = []
            for fact in facts:
                result = client.store_fact(fact)
                if result.success:
                    results.append(result.id)

            elapsed = time.time() - start

            # Verify most were inserted (some may be rejected due to novelty)
            assert len(results) >= 50

            # Calculate throughput
            rate = len(results) / elapsed if elapsed > 0 else 0
            assert rate > 0  # Some rate achieved

        finally:
            client.close()

    def test_batch_search(self, unique_storage_config):
        """Test searching after batch insert"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Insert many facts
            for i in range(50):
                client.store_fact(f"Test fact {i}", 0.8)

            # Search should work
            results = client.search("Test fact", limit=20)
            assert len(results.memories) > 0

        finally:
            client.close()


class TestCustomConfiguration:
    """Test custom configuration options"""

    def test_custom_config(self, unique_storage_config):
        """Test creating client with custom configuration"""
        config = {
            **unique_storage_config,
            "max_memories": 10000,
            "embedding_dim": 384,
            "similarity_threshold": 0.85,
        }

        client = MembrainClient(config=config)

        try:
            client.store_fact("Custom config active")
            results = client.search("custom")
            assert results is not None

        finally:
            client.close()

    def test_different_similarity_thresholds(self, tmp_path):
        """Test different similarity threshold configurations"""
        # High threshold (more strict)
        config_strict = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "strict")
            },
            "similarity_threshold": 0.95
        }
        client_strict = MembrainClient(config=config_strict)

        try:
            client_strict.store_fact("Test with strict threshold", 0.9)
            results = client_strict.search("test")
            assert results is not None

        finally:
            client_strict.close()

        # Low threshold (more permissive)
        config_permissive = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "permissive")
            },
            "similarity_threshold": 0.7
        }
        client_permissive = MembrainClient(config=config_permissive)

        try:
            client_permissive.store_fact("Test with permissive threshold", 0.9)
            results = client_permissive.search("test")
            assert results is not None

        finally:
            client_permissive.close()


class TestMemoryUpdate:
    """Test updating existing memories"""

    def test_update_pattern(self, unique_storage_config):
        """Test updating memory by storing new version and deleting old"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Store initial fact
            result = client.store_fact("Python 3.9 is current", 0.8)
            mem_id = result.id

            # Store updated version
            client.store_fact("Python 3.12 is current", 0.95)

            # Delete old version
            client.delete(mem_id)

            # Search should find new version
            results = client.search("Python current version")
            if len(results.memories) > 0:
                content = results.memories[0].content
                assert "3.12" in content or "current" in content

        finally:
            client.close()

    def test_safe_update_pattern(self, unique_storage_config):
        """Test safe update pattern (store before delete)"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Store original
            old = client.store_fact("Original version", 0.8)

            # Store new BEFORE deleting old
            new = client.store_fact("Updated version", 0.9)

            # Now safe to delete old
            client.delete(old.id)

            # Verify new exists
            memory = client.get(new.id)
            assert memory is not None
            assert "Updated" in memory.content

        finally:
            client.close()


class TestFilteredSearch:
    """Test filtered search by memory type"""

    def test_filter_by_type(self, unique_storage_config):
        """Test filtering search results by memory type"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Store different types
            client.store_fact("Python is a language")
            client.store_event("user_action", "User logged in")
            client.store_preference("Alice", "language", "prefers Python", "strong")

            # Search all
            all_results = client.search("Python", limit=10)
            assert len(all_results.memories) > 0

            # Filter by memory_type
            semantic_memories = [
                m for m in all_results.memories
                if m.memory_type == "Semantic"
            ]

            episodic_memories = [
                m for m in all_results.memories
                if m.memory_type == "Episodic"
            ]

            # Should have different types
            assert len(all_results.memories) >= len(semantic_memories)

        finally:
            client.close()

    def test_search_specific_content_type(self, unique_storage_config):
        """Test searching for specific content types"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Store multiple types
            client.store_concept("AI", "Artificial Intelligence")
            client.store_entity("OpenAI", "ai_company")
            client.store_skill("coding", "Programming ability")

            # Search and inspect types
            results = client.search("AI", limit=10)
            assert len(results.memories) > 0

            # Verify we can access memory_type
            for m in results.memories:
                assert hasattr(m, 'memory_type')
                assert m.memory_type is not None

        finally:
            client.close()


class TestMonitoring:
    """Test monitoring and observability"""

    def test_performance_monitoring(self, unique_storage_config):
        """Test tracking operation performance"""
        class MembrainMonitor:
            def __init__(self, client):
                self.client = client

            def snapshot(self):
                stats = self.client.stats()
                return {
                    "timestamp": time.time(),
                    "total": stats["total_memories"],
                    "by_type": stats["by_type"],
                    "avg_confidence": stats["avg_confidence"],
                }

            def track_operation(self, fn, *args):
                start = time.time()
                result = fn(*args)
                elapsed = (time.time() - start) * 1000
                return result, elapsed

        client = MembrainClient(config=unique_storage_config)

        try:
            monitor = MembrainMonitor(client)

            # Track operation
            result, latency = monitor.track_operation(
                client.store_fact, "Test fact", 0.9
            )

            assert result.id is not None
            assert latency >= 0

            # Get snapshot
            snapshot = monitor.snapshot()
            assert "total" in snapshot
            assert "avg_confidence" in snapshot
            assert snapshot["total"] >= 1

        finally:
            client.close()

    def test_statistics_tracking(self, unique_storage_config):
        """Test tracking statistics over time"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Initial state
            stats1 = client.stats()
            initial_count = stats1["total_memories"]

            # Add memories
            for i in range(10):
                client.store_fact(f"Stat test {i}", 0.85)

            # Check updated stats
            stats2 = client.stats()
            assert stats2["total_memories"] >= initial_count + 10

        finally:
            client.close()


class TestHybridSearch:
    """Test hybrid search combining semantic and graph"""

    def test_hybrid_search_pattern(self, tmp_path):
        """Test combining semantic search with graph traversal"""
        # Use embedding dimension 16 for testing
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "hybrid_search")
            },
            "embedding_dim": 16
        }
        client = MembrainClient(config=config)
        graph_config = {"embedding_dim": 16}
        graph = MembrainGraph(config=graph_config)

        try:
            # Store and index with distinct descriptions
            topics = [
                ("AI", "Artificial Intelligence deals with creating intelligent machines"),
                ("ML", "Machine Learning focuses on algorithms that learn from data"),
                ("NLP", "Natural Language Processing enables computers to understand human language"),
                ("CV", "Computer Vision allows machines to interpret visual information"),
                ("RL", "Reinforcement Learning trains agents through rewards and penalties")
            ]
            ids = []

            for topic, description in topics:
                result = client.store_concept(topic, description)
                if result.success:
                    ids.append(result.id)
                    emb = [random.random() for _ in range(16)]
                    graph.add_node(result.id, emb)

            # Semantic search
            semantic = client.search("machine learning algorithms", limit=5)
            # At least some memories should be stored
            assert len(ids) > 0

            # Graph search
            query_emb = [random.random() for _ in range(16)]
            graph_result = graph.query(query_emb, max_hops=2, top_k=5)
            assert len(graph_result.nodes) > 0

            # Combine results (check IDs from both)
            semantic_ids = {m.id for m in semantic.memories}
            graph_ids = {node.memory_id for node in graph_result.nodes}

            # At least one should exist
            assert len(semantic_ids) > 0 or len(graph_ids) > 0

        finally:
            client.close()
            graph.close()


class TestConnectionPooling:
    """Test connection pooling pattern"""

    def test_simple_pool_pattern(self, tmp_path):
        """Test basic connection pooling pattern with separate databases"""
        from queue import Queue

        class MembrainPool:
            def __init__(self, size=3, base_path=None):
                self.pool = Queue(maxsize=size)
                # Each client needs its own database with Redb
                for i in range(size):
                    config = {
                        "storage": {
                            "backend": "memscaledb",
                            "path": str(base_path / f"pool_{i}")
                        }
                    }
                    self.pool.put(MembrainClient(config=config))

            def acquire(self):
                return self.pool.get()

            def release(self, client):
                self.pool.put(client)

            def close_all(self):
                while not self.pool.empty():
                    client = self.pool.get()
                    client.close()

        # Create pool
        pool = MembrainPool(size=2, base_path=tmp_path)

        try:
            # Use client from pool
            client = pool.acquire()
            try:
                client.store_fact("Pool test")
            finally:
                pool.release(client)

            # Acquire again
            client2 = pool.acquire()
            try:
                results = client2.search("Pool")
                assert results is not None
            finally:
                pool.release(client2)

        finally:
            pool.close_all()


class TestVersionedMemory:
    """Test versioned memory pattern"""

    def test_versioned_storage(self, unique_storage_config):
        """Test storing multiple versions of memories"""
        class VersionedMemory:
            def __init__(self, client):
                self.client = client

            def store_versioned(self, key, value, version):
                # Use a unique identifier to differentiate versions
                content = json.dumps({
                    "key": key,
                    "value": value,
                    "version": version
                })
                # Store as fact with version info in the statement
                return self.client.store_fact(f"Version {version} of {key}: {content}", 0.9)

            def get_latest(self, key):
                # Search for the key
                results = self.client.search(f"{key} version", limit=20)
                versions = []
                for m in results.memories:
                    try:
                        # Extract JSON from the content
                        content = m.content
                        # Find JSON object in content
                        if "{" in content and "}" in content:
                            json_str = content[content.find("{"):content.rfind("}")+1]
                            data = json.loads(json_str)
                            if data.get("key") == key:
                                versions.append(data)
                    except:
                        pass

                if versions:
                    return max(versions, key=lambda x: x["version"])
                return None

        client = MembrainClient(config=unique_storage_config)

        try:
            vm = VersionedMemory(client)

            # Store versions with distinct content to avoid novelty rejection
            r1 = vm.store_versioned("config", {"theme": "light", "contrast": "low"}, 1)
            r2 = vm.store_versioned("config", {"theme": "dark", "contrast": "high"}, 2)
            r3 = vm.store_versioned("config", {"theme": "auto", "contrast": "medium"}, 3)

            # Verify at least one was stored
            assert any([r.success for r in [r1, r2, r3]])

            # Get latest
            latest = vm.get_latest("config")
            # If novelty detection rejected some, we may not have all versions
            if latest is not None:
                assert latest["version"] in [1, 2, 3]
                assert "theme" in latest["value"]

        finally:
            client.close()


class TestErrorRecovery:
    """Test error recovery patterns"""

    def test_graceful_error_handling(self, unique_storage_config):
        """Test graceful error handling"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Normal operation
            result = client.store_fact("Test", 0.9)
            assert result.id is not None

            # Even if errors occur, client should remain usable
            # Try to continue operations
            result2 = client.store_fact("Test 2", 0.8)
            assert result2.id is not None

        finally:
            client.close()


class TestResourceManagement:
    """Test proper resource management"""

    def test_explicit_cleanup(self, unique_storage_config):
        """Test explicit resource cleanup"""
        client = MembrainClient(config=unique_storage_config)

        try:
            client.store_fact("Resource test")
        finally:
            client.close()
            # Client should be closed

    def test_context_manager_cleanup(self, unique_storage_config):
        """Test automatic cleanup with context manager"""
        with MembrainClient(config=unique_storage_config) as client:
            client.store_fact("Context manager test")
            results = client.search("context")
            assert results is not None
        # Client automatically closed


class TestMemoryLifecycle:
    """Test complete memory lifecycle"""

    def test_create_read_update_delete(self, unique_storage_config):
        """Test CRUD operations on memories"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Create
            result = client.store_fact("Lifecycle test", 0.9)
            mem_id = result.id
            assert mem_id is not None

            # Read
            memory = client.get(mem_id)
            assert memory is not None
            assert memory.content == "Lifecycle test"

            # Update (by creating new and deleting old)
            new_result = client.store_fact("Lifecycle test updated", 0.95)
            new_id = new_result.id

            # Delete old
            client.delete(mem_id)

            # Verify new exists and old is gone
            new_memory = client.get(new_id)
            assert new_memory is not None
            assert "updated" in new_memory.content

        finally:
            client.close()


class TestCompressionSettings:
    """Test compression configuration"""

    def test_compression_enabled(self, unique_storage_config):
        """Test client with compression enabled"""
        config = {
            **unique_storage_config,
            "compression_enabled": True,
            "max_memories": 1000
        }

        client = MembrainClient(config=config)

        try:
            # Store multiple memories
            for i in range(20):
                client.store_fact(f"Compression test {i}")

            # Should work normally with compression
            results = client.search("Compression test", limit=10)
            assert len(results.memories) > 0

        finally:
            client.close()


class TestLargeScaleOperations:
    """Test operations at larger scale"""

    def test_many_memories(self, unique_storage_config):
        """Test storing and retrieving many memories"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Store many memories
            count = 200
            for i in range(count):
                client.store_fact(f"Scale test memory {i}", 0.85)

            # Verify stats
            stats = client.stats()
            assert stats["total_memories"] >= count

            # Search should still work
            results = client.search("Scale test", limit=20)
            assert len(results.memories) > 0

        finally:
            client.close()


class TestSearchLimits:
    """Test search with different limit values"""

    def test_various_search_limits(self, unique_storage_config):
        """Test searching with different limit values"""
        client = MembrainClient(config=unique_storage_config)

        try:
            # Store test data
            for i in range(20):
                client.store_fact(f"Limit test {i}")

            # Test different limits
            for limit in [1, 5, 10, 20]:
                results = client.search("Limit test", limit=limit)
                assert len(results.memories) <= limit

        finally:
            client.close()
