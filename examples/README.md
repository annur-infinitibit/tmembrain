# Examples

Working examples demonstrating all Membrain Python and JavaScript bindings.

## Python Examples

All Python examples are in `python/` and import from the `membrain` package.

```bash
export MEMBRAIN_LIB_PATH=/path/to/libmembrain_ffi.so
python examples/python/client_example.py
```

### Memory Layer (Membrain)

| Example | Description |
|---------|-------------|
| [conversation_example.py](python/conversation_example.py) | Automatic conversation management: memory retrieval, extraction, and case-based learning with any LLM |
| [client_example.py](python/client_example.py) | Store and search all memory types: facts, preferences, events, observations, concepts, entities, workflows, skills, patterns, goals, tasks, and case-based reasoning |
| [graph_example.py](python/graph_example.py) | Graph-based memory with multi-hop traversal, pruning, and persistence |
| [sharded_index_example.py](python/sharded_index_example.py) | HNSW shards with k-means routing, rebalancing, and persistence |
| [multi_tenant_example.py](python/multi_tenant_example.py) | Isolated per-tenant vector indices with tenant management |
| [reranker_example.py](python/reranker_example.py) | LLM-based reranking with Anthropic, OpenAI, Cohere, and Jina |

### Vector Indices (MemscaleDB)

| Example | Index Type | Description |
|---------|-----------|-------------|
| [flat_index_example.py](python/flat_index_example.py) | Flat | Brute-force exact search, 100% recall |
| [hnsw_index_example.py](python/hnsw_index_example.py) | HNSW | Hierarchical graph-based ANN with save/load |
| [ivf_index_example.py](python/ivf_index_example.py) | IVF | K-means partitioned search, built from training data |
| [lsh_index_example.py](python/lsh_index_example.py) | LSH | Random hyperplane hashing |
| [vamana_index_example.py](python/vamana_index_example.py) | Vamana | DiskANN-style graph, built from training data |
| [concurrent_indices_example.py](python/concurrent_indices_example.py) | All concurrent | Thread-safe variants with multi-threaded demos |

## JavaScript Examples

All JavaScript examples are in `javascript/` and import from the `membrain` package.

```bash
export MEMBRAIN_LIB_PATH=/path/to/libmembrain_ffi.so
node examples/javascript/client_example.mjs
```

### Memory Layer (Membrain)

| Example | Description |
|---------|-------------|
| [conversation_example.mjs](javascript/conversation_example.mjs) | Automatic conversation management: memory retrieval, extraction, and case-based learning with any LLM |
| [client_example.mjs](javascript/client_example.mjs) | Store and search all memory types with case-based reasoning |
| [graph_example.mjs](javascript/graph_example.mjs) | Graph-based memory with multi-hop traversal, pruning, and persistence |
| [sharded_index_example.mjs](javascript/sharded_index_example.mjs) | HNSW shards with k-means routing, rebalancing, and persistence |
| [multi_tenant_example.mjs](javascript/multi_tenant_example.mjs) | Isolated per-tenant vector indices with tenant management |
| [reranker_example.mjs](javascript/reranker_example.mjs) | LLM-based reranking with Anthropic, OpenAI, Cohere, and Jina |

### Vector Indices (MemscaleDB)

| Example | Index Type | Description |
|---------|-----------|-------------|
| [flat_index_example.mjs](javascript/flat_index_example.mjs) | Flat | Brute-force exact search, 100% recall |
| [hnsw_index_example.mjs](javascript/hnsw_index_example.mjs) | HNSW | Hierarchical graph-based ANN with save/load |
| [ivf_index_example.mjs](javascript/ivf_index_example.mjs) | IVF | K-means partitioned search, built from training data |
| [lsh_index_example.mjs](javascript/lsh_index_example.mjs) | LSH | Random hyperplane hashing |
| [vamana_index_example.mjs](javascript/vamana_index_example.mjs) | Vamana | DiskANN-style graph, built from training data |
| [concurrent_indices_example.mjs](javascript/concurrent_indices_example.mjs) | All concurrent | Thread-safe variants (Flat, HNSW, IVF, LSH, Vamana) |

## Integration Examples

Framework integration examples are in `integrations/`.

| Example | Description |
|---------|-------------|
| [test_direct.py](integrations/test_direct.py) | Direct MembrainClient usage |
| [langchain_example.py](integrations/langchain_example.py) | LangChain integration |
| [llamaindex_example.py](integrations/llamaindex_example.py) | LlamaIndex integration |
| [autogen_example.py](integrations/autogen_example.py) | AutoGen integration |
