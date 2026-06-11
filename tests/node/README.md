# Membrain Node.js Tests

Test suite for Membrain Node.js/JavaScript bindings based on the official documentation.

## Test Coverage

### test_basic_usage.test.js
Tests from `docs/cookbooks/basic-usage.mdx`:
- Fact storage and retrieval
- Event logging
- User preferences
- Entity management
- Workflow documentation
- Skill registry
- Statistics
- Memory retrieval by ID
- Error handling
- Custom configuration

### test_graph_memory.test.js
Tests from `docs/cookbooks/graph-memory.mdx`:
- Graph creation with configuration
- Node operations
- Single-hop and multi-hop queries
- Graph persistence (save/load)
- Graph pruning
- Client-Graph integration
- Edge operations
- Scalability
- Custom embedding dimensions

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

3. **Install Node.js dependencies:**
   ```bash
   cd tests/node
   npm install
   ```

4. **Install Membrain Node package:**
   ```bash
   cd /home/yeahiasarker/Documents/personal/membrain/membrain-node
   npm install
   npm link
   cd /home/yeahiasarker/Documents/personal/membrain/tests/node
   npm link membrain
   ```

## Environment Variables

Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running Tests

### Run all tests:
```bash
npm test
```

### Run specific test file:
```bash
npm run test:basic
npm run test:graph
```

### Run with watch mode:
```bash
npm run test:watch
```

### Run with coverage:
```bash
npm run test:coverage
```

### Run using Jest directly:
```bash
npx jest test_basic_usage.test.js
npx jest test_graph_memory.test.js
```

## Test Organization

- **Jest framework**: Using Jest for testing (familiar to JavaScript developers)
- **Descriptive test names**: Clear test descriptions
- **Proper cleanup**: All tests properly close resources
- **Skip mechanism**: Tests automatically skip if Membrain is not available

## Notes

- Tests will be skipped if the Membrain module is not found
- Tests use `try/finally` blocks to ensure proper resource cleanup
- Random embeddings are used for graph operations (use real embedding models in production)
- MembrainClient is not thread-safe

## Continuous Integration

These tests can be integrated into CI/CD pipelines:
```bash
npm test -- --ci --coverage --maxWorkers=2
```

## Contributing

When adding new tests:
1. Follow the existing naming conventions (`test_*.test.js`)
2. Use descriptive test names
3. Ensure proper resource cleanup with try/finally
4. Update this README if adding new test files
