# Membrain Test Suite

Comprehensive test suite for Membrain covering all examples from the official documentation cookbooks.

## Overview

This test suite validates the Membrain memory system across multiple language bindings:
- **Python** (`tests/python/`) - Comprehensive tests using pytest
- **Node.js** (`tests/node/`) - JavaScript tests using Jest
- **Rust** (`tests/rust/`) - Native Rust tests (coming soon)

All tests are based on real-world examples from the documentation, ensuring that the examples work as documented.

## Quick Start

### 1. Build Membrain

```bash
# From project root
cargo build --release
```

### 2. Set Library Path

```bash
export MEMBRAIN_LIB_PATH=$(pwd)/target/release/libmembrain_ffi.so
```

### 3. Run Tests

#### Python Tests
```bash
cd tests/python
pip install -r requirements.txt
pip install -e ../../membrain-py
pytest -v
```

#### Node.js Tests
```bash
cd tests/node
npm install
npm test
```

## Test Coverage by Documentation

### Quickstart Guide (`docs/quickstart.mdx`)
- ✅ Basic LLM integration with OpenAI
- ✅ Memory types (Semantic, Episodic, Procedural, Agent State)
- ✅ Searching for LLM context
- ✅ Graph memory for complex reasoning
- ✅ Configuration options
- ✅ Error handling

### Basic Usage Cookbook (`docs/cookbooks/basic-usage.mdx`)
- ✅ RAG system with OpenAI
- ✅ Chatbot with conversation memory
- ✅ Fact storage and retrieval
- ✅ User preference system
- ✅ Event logging
- ✅ Entity management
- ✅ Workflow documentation
- ✅ Skill registry
- ✅ Statistics monitoring

### Graph Memory Cookbook (`docs/cookbooks/graph-memory.mdx`)
- ✅ LLM knowledge graph setup
- ✅ Multi-hop queries for complex reasoning
- ✅ Graph persistence (save/load)
- ✅ Graph pruning
- ✅ Client-Graph integration
- ✅ Embedding model integration patterns

### Multi-Agent Cookbook (`docs/cookbooks/multi-agent.mdx`)
- ✅ Shared knowledge base for LLM agents
- ✅ Agent skill registry
- ✅ Multi-agent coordination
- ✅ Agent state tracking
- ✅ Hierarchical agent systems
- ✅ Collaborative learning

### Advanced Patterns Cookbook (`docs/cookbooks/advanced-patterns.mdx`)
- ✅ Memory deduplication
- ✅ Batch operations
- ✅ Custom configuration
- ✅ Memory update patterns
- ✅ Filtered search by type
- ✅ Monitoring and observability
- ✅ Hybrid search (semantic + graph)
- ✅ Connection pooling
- ✅ Versioned memory

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Required for OpenAI integration tests
OPENAI_API_KEY=sk-...

# Optional: Custom library path
MEMBRAIN_LIB_PATH=/path/to/libmembrain_ffi.so
```

### Python Environment

```bash
cd tests/python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ../../membrain-py
```

### Node.js Environment

```bash
cd tests/node
npm install
cd ../../membrain-node
npm link
cd ../tests/node
npm link membrain
```

## Test Organization

### Python Tests (`tests/python/`)

```
tests/python/
├── test_basic_usage.py         # Basic operations and LLM integration
├── test_graph_memory.py        # Graph-based memory operations
├── test_multi_agent.py         # Multi-agent systems
├── test_advanced_patterns.py   # Advanced patterns and optimization
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── run_tests.sh               # Test runner script
└── README.md                   # Python-specific documentation
```

### Node.js Tests (`tests/node/`)

```
tests/node/
├── test_basic_usage.test.js    # Basic operations
├── test_graph_memory.test.js   # Graph operations
├── package.json                # Node dependencies and scripts
└── README.md                   # Node-specific documentation
```

## Running Tests

### All Tests (Python)

```bash
cd tests/python
./run_tests.sh all
```

### Specific Test Categories

```bash
# Python
cd tests/python
./run_tests.sh basic      # Basic usage tests
./run_tests.sh graph      # Graph memory tests
./run_tests.sh multiagent # Multi-agent tests
./run_tests.sh advanced   # Advanced patterns
./run_tests.sh quick      # Skip OpenAI tests

# Node.js
cd tests/node
npm run test:basic
npm run test:graph
```

### Individual Test Files

```bash
# Python
pytest test_basic_usage.py -v
pytest test_graph_memory.py::TestGraphBasics -v

# Node.js
npx jest test_basic_usage.test.js
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Membrain
        run: cargo build --release

      - name: Run Python Tests
        run: |
          cd tests/python
          pip install -r requirements.txt
          pytest -v

      - name: Run Node.js Tests
        run: |
          cd tests/node
          npm install
          npm test
```

## Test Guidelines

### Writing New Tests

1. **Follow documentation examples**: Tests should mirror cookbook examples
2. **Proper resource cleanup**: Always use try/finally or context managers
3. **Descriptive names**: Test names should clearly describe what they test
4. **Documentation**: Add docstrings/comments explaining test purpose
5. **Independence**: Tests should not depend on other tests
6. **Error handling**: Test both success and failure cases

### Example Test Structure (Python)

```python
class TestFeature:
    """Test a specific feature from documentation"""

    def test_basic_operation(self):
        """Test basic operation as shown in docs"""
        client = MembrainClient()
        try:
            # Test implementation
            result = client.store_fact("test", 0.9)
            assert result.id is not None
        finally:
            client.close()
```

### Example Test Structure (JavaScript)

```javascript
describe('Feature Tests', () => {
  test('should perform basic operation', () => {
    const client = new MembrainClient();
    try {
      const result = client.storeFact("test", 0.9);
      expect(result.id).toBeDefined();
    } finally {
      client.close();
    }
  });
});
```

## Performance Benchmarks

Some tests include basic performance benchmarks:
- Batch insertion throughput
- Search latency
- Graph query performance

Run with verbose output to see performance metrics:
```bash
pytest -v -s
```

## Troubleshooting

### Common Issues

1. **Library not found**
   ```
   Error: libmembrain_ffi.so not found
   ```
   Solution: Set `MEMBRAIN_LIB_PATH` or rebuild with `cargo build --release`

2. **OpenAI tests skipped**
   ```
   SKIPPED [1] test requires OPENAI_API_KEY
   ```
   Solution: Set `OPENAI_API_KEY` in `.env` file

3. **Import errors (Python)**
   ```
   ImportError: cannot import name 'MembrainClient'
   ```
   Solution: Install membrain package with `pip install -e ./membrain-py`

4. **Module not found (Node.js)**
   ```
   Cannot find module 'membrain'
   ```
   Solution: Link the package with `npm link` from membrain-node directory

## Contributing

When contributing tests:
1. Ensure tests are based on documented examples
2. Add tests for new documentation examples
3. Update README files when adding new test categories
4. Ensure tests pass locally before submitting
5. Follow existing code style and patterns
