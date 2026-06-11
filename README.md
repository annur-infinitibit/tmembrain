<h1 align="center">Membrain</h1>

<p align="center">A high-performance memory layer for Large Language Models (LLMs) with semantic search, graph-based relationships, and multi-agent coordination. Give your LLM applications long-term memory, context retention, and knowledge persistence.</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75+-orange?logo=rust" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/node.js-18+-green?logo=node.js&logoColor=white" alt="Node.js 18+">
  <img src="https://img.shields.io/badge/version-0.1.0-brightgreen" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/status-alpha-yellow" alt="Status: Alpha">
</p>

## Features

- **Automatic Conversation Management**: Drop-in conversation loop that retrieves relevant memories, calls your LLM, extracts facts/preferences/entities from each turn, and learns from outcomes -- zero manual memory plumbing
- **LLM-Optimized Memory**: Store and retrieve conversation context, facts, and knowledge for LLMs
- **Semantic Search**: Fast vector similarity search with automatic deduplication - perfect for RAG systems
- **Metadata Filtering**: Organize and filter memories by custom metadata (tags, categories, session IDs, etc.)
- **Multi-layered Memory**: Episodic (conversations), semantic (facts), procedural (workflows), and agent state
- **Case-Based Reasoning**: Store conversation outcomes with reward signals and retrieve past successes/failures to guide future responses
- **Graph Memory Layer**: Multi-hop traversal for complex reasoning and relationship queries
- **FFI Support**: Python and JavaScript bindings - integrate with any LLM framework
- **Production Ready**: Compression, audit trails, background jobs for enterprise LLM applications

## Quick Start

### Python

```python
from membrain import Conversation
from openai import OpenAI

openai_client = OpenAI()

def llm(messages):
    response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

with Conversation(llm_callable=llm) as conv:
    # Membrain automatically retrieves relevant memories, calls your LLM,
    # and extracts facts, preferences, entities from each turn
    response = conv.reply("I prefer dark mode and work at Acme Corp as a backend engineer.")
    response = conv.reply("Can you recommend a code editor setup for me?")

    # Memories persist -- ask across turns or even across sessions
    response = conv.reply("What do you remember about me?")

    # Store the outcome so future conversations learn from this one
    conv.end(outcome="User was satisfied with recommendations", reward=1.0)
```

### JavaScript

```javascript
import { Conversation } from "membrain";
import OpenAI from "openai";

const openai = new OpenAI();

async function llm(messages) {
  const response = await openai.chat.completions.create({ model: "gpt-4o-mini", messages });
  return response.choices[0].message.content;
}

const conv = new Conversation(llm);

try {
  const r1 = await conv.reply("I prefer dark mode and work at Acme Corp as a backend engineer.");
  const r2 = await conv.reply("Can you recommend a code editor setup for me?");
  const r3 = await conv.reply("What do you remember about me?");

  conv.end("User was satisfied with recommendations", 1.0);
} finally {
  conv.close();
}
```

### Manual Memory Control

For fine-grained control, use `MembrainClient` directly:

```python
from membrain import MembrainClient

memory = MembrainClient()

memory.store_fact("Python PEP 8 is the style guide", confidence=0.95)
memory.store_preference(holder="user", subject="theme", preference="dark mode", strength="strong")
memory.store_entity(name="Acme Corp", entity_type="organization")

results = memory.search("coding standards", limit=5)
for m in results.memories:
    print(f"[{m.memory_type}] {m.content}")

memory.close()
```

## How Automatic Conversation Management Works

Each call to `conv.reply()` runs a full memory-augmented loop:

1. **Retrieve** -- Semantic search finds relevant memories and past conversation cases
2. **Augment** -- Memories and case-based reasoning context are injected into the system prompt
3. **Generate** -- Your LLM is called with the enriched prompt and conversation history
4. **Extract** -- The turn is analyzed to extract facts, preferences, observations, entities, and concepts
5. **Store** -- Extracted memories are persisted for future retrieval across sessions

When you call `conv.end()`, the full conversation is stored as a case with a reward signal. Future conversations retrieve successful (and unsuccessful) cases to inform response strategy.

| Parameter | Default | Description |
|---|---|---|
| `llm_callable` | required | Function that takes messages and returns a response string |
| `system_prompt` | built-in | Custom system prompt |
| `memory_limit` | 10 | Max memories injected per turn |
| `auto_extract` | true | Automatically extract and store memories |
| `history_limit` | 50 | Max turns kept in the context window |

## Installation

### Python

```bash
pip install membrain
```

Set `MEMBRAIN_LIB_PATH` to point to `libmembrain_ffi.so` if not auto-detected.

### JavaScript

```bash
npm install membrain
```

Set `MEMBRAIN_LIB_PATH` to point to the shared library if needed.

## Documentation

**[Complete Documentation](docs/)** - Full documentation index

**Quick Links:**
- [Quickstart Guide](docs/quickstart.md) - Get started in 5 minutes
- [Python API Reference](docs/python-api.md) - Complete Python API
- [JavaScript API Reference](docs/javascript-api.md) - Complete JavaScript/TypeScript API
- [Architecture Guide](docs/architecture.md) - System design and internals
- [Cookbooks](docs/cookbooks/) - Practical examples
  - [Basic Usage](docs/cookbooks/basic-usage.md) - Fundamental operations
  - [Graph Memory](docs/cookbooks/graph-memory.md) - Multi-hop traversal
  - [Multi-Agent Systems](docs/cookbooks/multi-agent.md) - Collaborative agents
  - [Advanced Patterns](docs/cookbooks/advanced-patterns.md) - Optimization techniques

## Building from Source

```bash
# Build Rust core
cargo build --release

# The FFI library will be at target/release/libmembrain_ffi.so
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
