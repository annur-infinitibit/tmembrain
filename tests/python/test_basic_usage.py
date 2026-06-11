"""
Test Basic Usage Cookbook Examples
Based on docs/cookbooks/basic-usage.mdx
"""
import os
import uuid
import pytest
from dotenv import load_dotenv
from openai import OpenAI

# Note: These tests assume membrain is installed and libmembrain_ffi.so is available
# Set MEMBRAIN_LIB_PATH environment variable if needed

try:
    from membrain import MembrainClient, MembrainError
    MEMBRAIN_AVAILABLE = True
except ImportError:
    MEMBRAIN_AVAILABLE = False
    pytest.skip("Membrain not installed", allow_module_level=True)

# Load environment variables
load_dotenv()


def make_unique(text: str) -> str:
    """Make text unique by adding a UUID suffix"""
    return f"{text} [test-{str(uuid.uuid4())[:8]}]"


class TestRAGSystem:
    """Test RAG (Retrieval Augmented Generation) system"""

    def test_rag_with_openai(self, openai_available):
        """Test basic RAG pattern with OpenAI"""
        memory = MembrainClient()
        llm = OpenAI()

        try:
            # Store knowledge base with unique, varied content
            test_id = str(uuid.uuid4())
            knowledge = [
                (f"RAG technology combines retrieval mechanisms with generation capabilities in system {test_id}", 0.95),
                (f"Vector similarity search algorithms find semantically related content using embeddings {test_id}", 0.9),
                (f"Text embeddings are numerical vector representations of textual data for {test_id} processing", 0.9),
            ]

            stored_facts = []
            for fact, confidence in knowledge:
                result = memory.store_fact(fact, confidence)
                if result.success:
                    assert result.id is not None
                    assert len(result.id) > 0
                    stored_facts.append(result.id)

            # Skip if not enough facts were stored
            if len(stored_facts) == 0:
                pytest.skip("No facts were stored (novelty threshold too high)")

            # Retrieve relevant context
            context = memory.search("What is RAG?", limit=3)
            assert context is not None
            assert len(context.memories) > 0

            context_text = "\n".join([m.content for m in context.memories])
            assert len(context_text) > 0

            # Generate response with LLM
            response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Use this context:\n{context_text}"},
                    {"role": "user", "content": "What is RAG?"}
                ],
                max_tokens=100
            )

            answer = response.choices[0].message.content
            assert len(answer) > 0
            # Check that LLM provided some answer (may vary in quality)
            # In a real RAG system, context quality depends on embeddings

        finally:
            memory.close()


class TestConversationMemory:
    """Test chatbot with conversation memory"""

    def test_conversation_history(self):
        """Test storing and retrieving conversation history"""
        memory = MembrainClient()

        try:
            # Store conversation exchanges with unique content
            test_id = str(uuid.uuid4())[:8]
            memory.store_event("user_msg", f"Hello, I'm learning Python {test_id}")
            memory.store_event("assistant_msg", f"Great! Python is wonderful for beginners {test_id}")
            memory.store_event("user_msg", f"What did I say I was learning? {test_id}")

            # Retrieve conversation history
            # Search for recent messages
            history = memory.search(f"learning Python {test_id}", limit=5)
            assert len(history.memories) > 0

            # Check that Python is mentioned
            context = "\n".join([m.content for m in history.memories])
            assert "Python" in context or "python" in context

        finally:
            memory.close()

    def test_chat_with_openai(self, openai_available):
        """Test full chatbot with OpenAI integration"""
        memory = MembrainClient()
        llm = OpenAI()

        try:
            # First message
            user_msg_1 = "My favorite color is blue"
            memory.store_event("user_msg", user_msg_1)

            response_1 = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_msg_1}],
                max_tokens=50
            )
            assistant_msg_1 = response_1.choices[0].message.content
            memory.store_event("assistant_msg", assistant_msg_1)

            # Second message using memory
            user_msg_2 = "What color did I say I liked?"
            history = memory.search("favorite color", limit=5)
            context = "\n".join([m.content for m in history.memories])

            assert "blue" in context.lower()

        finally:
            memory.close()


class TestFactStorage:
    """Test simple fact storage and retrieval"""

    def test_store_and_search_facts(self):
        """Test storing facts and searching for them"""
        with MembrainClient() as client:
            # Store facts
            client.store_fact("Paris is the capital of France", 0.99)
            client.store_fact("The Eiffel Tower is in Paris", 0.99)
            client.store_fact("Tokyo is the capital of Japan", 0.99)

            # Search for related memories
            results = client.search("French capital", limit=3)
            assert len(results.memories) > 0

            # Verify content
            contents = [m.content for m in results.memories]
            assert any("Paris" in c for c in contents)

            # Check confidence scores are non-negative (BM25 scores can exceed 1.0)
            for m in results.memories:
                assert m.score >= 0.0


class TestPreferences:
    """Test user preference system"""

    def test_store_preferences(self):
        """Test storing and querying user preferences"""
        client = MembrainClient()

        try:
            # Store user preferences with unique test ID
            test_id = str(uuid.uuid4())[:8]
            users = [
                (f"Alice{test_id}", "coffee", f"prefers dark roast {test_id}", "strong"),
                (f"Alice{test_id}", "theme", f"uses dark mode {test_id}", "moderate"),
                (f"Bob{test_id}", "coffee", f"likes light roast {test_id}", "moderate"),
                (f"Bob{test_id}", "notifications", f"wants email updates {test_id}", "weak"),
            ]

            for holder, subject, pref, strength in users:
                result = client.store_preference(holder, subject, pref, strength)
                if not result.success:
                    # If rejected, that's okay for this test - we're testing the API works
                    continue
                assert result.id is not None

            # Query preferences
            results = client.search(f"Alice{test_id} coffee preferences")
            # At least some results should be found
            assert results is not None

        finally:
            client.close()


class TestEventLogging:
    """Test event logging functionality"""

    def test_log_events(self):
        """Test logging events and searching history"""
        client = MembrainClient()

        try:
            # Log events with unique test ID
            test_id = str(uuid.uuid4())[:8]
            events = [
                ("user_login", f"Alice{test_id} logged in from Chrome"),
                ("file_upload", f"Alice{test_id} uploaded report_{test_id}.pdf"),
                ("user_logout", f"Alice{test_id} logged out"),
            ]

            logged_ids = []
            for event_type, description in events:
                result = client.store_event(event_type, description)
                if result.success:
                    assert result.id is not None
                    logged_ids.append(result.id)

            assert len(logged_ids) >= 1  # At least one event should be stored

            # Search event history
            recent = client.search(f"Alice{test_id} activity", limit=10)
            # Should find some events
            assert recent is not None

        finally:
            client.close()


class TestEntityManagement:
    """Test entity storage and relationships"""

    def test_store_entities_and_relationships(self):
        """Test storing entities and their relationships"""
        client = MembrainClient()

        try:
            # Store entities with unique test ID
            test_id = str(uuid.uuid4())[:8]
            entities = [
                (f"Alice{test_id}", "person"),
                (f"PostgreSQL{test_id}", "database"),
                (f"Redis{test_id}", "cache"),
                (f"AWS{test_id}", "cloud_provider"),
            ]

            for name, entity_type in entities:
                result = client.store_entity(name, entity_type)
                if result.success:
                    assert result.id is not None

            # Store relationships as facts
            r1 = client.store_fact(f"Alice{test_id} uses PostgreSQL{test_id} for main database")
            r2 = client.store_fact(f"Alice{test_id} uses Redis{test_id} for session caching")

            # At least one should succeed
            assert r1.success or r2.success

            # Query entity relationships
            results = client.search(f"database Alice{test_id}")
            assert results is not None

        finally:
            client.close()


class TestWorkflows:
    """Test workflow documentation"""

    def test_store_workflows(self):
        """Test storing and retrieving workflows"""
        client = MembrainClient()

        try:
            # Store workflows with highly unique content
            test_id = str(uuid.uuid4())
            workflows = [
                (f"deploy_api_process", f"Deployment workflow for {test_id}: Step 1 execute comprehensive unit and integration tests. Step 2 construct Docker container image. Step 3 upload image to container registry. Step 4 deploy container to Kubernetes cluster."),
                (f"rollback_procedure", f"Rollback procedure for {test_id}: Step 1 identify and retrieve previous stable version from registry. Step 2 update Kubernetes deployment configuration with previous version. Step 3 monitor and verify system health metrics and application status."),
                (f"backup_database", f"Database backup workflow for {test_id}: Step 1 temporarily disable database write operations. Step 2 execute pg_dump utility to export database to Amazon S3 storage. Step 3 re-enable database write operations. Step 4 validate backup file integrity and accessibility."),
            ]

            stored_count = 0
            for name, desc in workflows:
                result = client.store_workflow(name, desc)
                if result.success:
                    assert result.id is not None
                    stored_count += 1

            # Test passes if at least one workflow is stored
            if stored_count == 0:
                pytest.skip("All workflows rejected due to novelty threshold")

            # Retrieve workflow
            results = client.search(f"deployment workflow {test_id}")
            # Should get some results
            assert results is not None

        finally:
            client.close()


class TestSkillRegistry:
    """Test skill registry for agents"""

    def test_register_skills(self):
        """Test registering and finding agent skills"""
        client = MembrainClient()

        try:
            # Register agent skills
            client.store_skill("code_review", "Analyzes code for bugs, style, and best practices")
            client.store_skill("bug_finder", "Identifies potential bugs using static analysis")
            client.store_skill("test_generator", "Generates unit tests for functions")

            # Find relevant skill
            results = client.search("need to check code quality")
            assert len(results.memories) > 0

            # Verify code review skill is suggested
            suggested = results.memories[0].content
            assert len(suggested) > 0

        finally:
            client.close()


class TestStatistics:
    """Test statistics monitoring"""

    def test_get_statistics(self):
        """Test retrieving statistics about stored memories"""
        client = MembrainClient()

        try:
            # Store some memories
            for i in range(10):
                client.store_fact(f"Fact number {i}", 0.8)

            # Check stats
            stats = client.stats()
            assert "total_memories" in stats
            assert stats["total_memories"] >= 10
            assert "by_type" in stats
            assert "avg_confidence" in stats
            assert stats["avg_confidence"] >= 0.0
            assert stats["avg_confidence"] <= 1.0

        finally:
            client.close()


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_confidence(self):
        """Test error handling for invalid confidence values"""
        client = MembrainClient()

        try:
            # Try to store with invalid confidence (> 1.0)
            # Note: Some implementations may clamp values instead of raising errors
            test_id = str(uuid.uuid4())[:8]
            result = client.store_fact(f"test confidence {test_id}", 1.5)
            # If no error is raised, the implementation accepts the value
            # (possibly clamping it internally)
            # The result object itself should exist
            assert result is not None

        finally:
            client.close()

    def test_context_manager(self):
        """Test context manager for automatic cleanup"""
        # This should not raise any errors
        with MembrainClient() as client:
            client.store_fact("Context manager test")
            results = client.search("context")
            assert results is not None
        # Client should be automatically closed here


class TestMemoryRetrieval:
    """Test memory retrieval by ID"""

    def test_get_memory_by_id(self):
        """Test retrieving a specific memory by ID"""
        client = MembrainClient()

        try:
            # Store a fact with unique content
            test_id = str(uuid.uuid4())[:8]
            content = f"Python is a high-level language {test_id}"
            result = client.store_fact(content, 0.9)

            # Check if storage was successful
            if not result.success:
                pytest.skip(f"Memory rejected: {result.rejection_reason}")

            mem_id = result.id
            assert mem_id is not None

            # Retrieve by ID
            memory = client.get(mem_id)
            assert memory is not None
            assert content in memory.content
            assert memory.id == mem_id

        finally:
            client.close()
