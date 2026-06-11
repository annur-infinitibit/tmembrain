# Membrain Python Tests

Comprehensive test suite for Membrain Python bindings based on the official documentation.

## Test Coverage

### test_basic_usage.py
Tests from `docs/cookbooks/basic-usage.mdx`:
- RAG (Retrieval Augmented Generation) systems with OpenAI
- Chatbot with conversation memory
- Simple fact storage and retrieval
- User preference systems
- Event logging
- Entity management
- Workflow documentation
- Skill registry for agents
- Statistics monitoring
- Error handling

### test_graph_memory.py
Tests from `docs/cookbooks/graph-memory.mdx`:
- Graph creation and configuration
- Adding nodes to the graph
- Single-hop and multi-hop queries
- Graph persistence (save/load)
- Graph pruning
- Integration between MembrainClient and MembrainGraph
- Edge operations
- Scalability testing

### test_multi_agent.py
Tests from `docs/cookbooks/multi-agent.mdx`:
- Shared knowledge base for multiple LLM agents
- Agent skill registry
- Multi-agent coordination
- Agent state tracking (goals, tasks, patterns)
- Hierarchical agent systems (manager/worker)
- Collaborative learning
- Agent communication patterns
- Team metrics

### test_advanced_patterns.py
Tests from `docs/cookbooks/advanced-patterns.mdx`:
- Automatic memory deduplication
- Batch operations and performance
- Custom configuration
- Memory update patterns
- Filtered search by type
- Monitoring and observability
- Hybrid search (semantic + graph)
- Connection pooling
- Versioned memory
- Resource management
- CRUD operations

## Prerequisites

1. **Build the Membrain library:**
   ```bash
   cd /home/yeahiasarker/Documents/personal/membrain
   cargo build --release
   ```

2. **Set library path:**
   ```bash
   export MEMBRAIN_LIB_PATH=/home/yeahiasarker/Documents/personal/membrain/target/release/libmembrain_ffi.so
   ```

3. **Install Python dependencies:**
   ```bash
   cd tests/python
   pip install -r requirements.txt
   ```

4. **Install Membrain Python package:**
   ```bash
   cd /home/yeahiasarker/Documents/personal/membrain/membrain-py
   pip install -e .
   ```

## Environment Variables

Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running Tests

### Run all tests:
```bash
cd /home/yeahiasarker/Documents/personal/membrain/tests/python
pytest -v
```

### Run specific test file:
```bash
pytest test_basic_usage.py -v
pytest test_graph_memory.py -v
pytest test_multi_agent.py -v
pytest test_advanced_patterns.py -v
```

### Run specific test class:
```bash
pytest test_basic_usage.py::TestRAGSystem -v
```

### Run specific test:
```bash
pytest test_basic_usage.py::TestRAGSystem::test_rag_with_openai -v
```

### Run tests with output:
```bash
pytest -v -s
```

### Run tests that don't require OpenAI:
```bash
pytest -v -m "not openai"
```

## Test Organization

Each test file follows this structure:
- **Class-based organization**: Related tests grouped into classes
- **Descriptive names**: Test names clearly describe what they test
- **Documentation**: Each test has a docstring explaining its purpose
- **Proper cleanup**: All tests properly close resources using try/finally or context managers

## Notes

- Tests that require OpenAI API will be skipped if `OPENAI_API_KEY` is not set
- Tests assume the Membrain library is properly built and accessible
- Some tests use random embeddings for graph operations (in production, use real embedding models)
- Connection pooling tests demonstrate patterns but don't test threading (MembrainClient is not thread-safe)

## Continuous Integration

These tests can be integrated into CI/CD pipelines:
```bash
# Example CI command
pytest --tb=short --junitxml=test-results.xml
```

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Add docstrings to explain what the test does
3. Ensure proper resource cleanup
4. Update this README if adding new test files
