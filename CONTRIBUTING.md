# Contributing to Membrain

Thank you for your interest in contributing to Membrain! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Prerequisites

- **Rust 1.75+** - Install via [rustup](https://rustup.rs/): `rustup update stable`
- **Python 3.10+** - For Python bindings development
- **Node.js 18+** - For JavaScript bindings development
- **Git** - For version control

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/membrain.git
cd membrain
```

### 2. Build the Project

```bash
# Build all crates (debug by default — release is only for perf work)
cargo build

# Build only the FFI library
cargo build -p membrain-ffi
```

Toolchain is pinned via `rust-toolchain.toml` (stable channel). Formatting
config lives in `rustfmt.toml`; workspace-wide lint rules are under
`[workspace.lints]` in root `Cargo.toml`.

The shared library will be at:
- Linux: `target/debug/libmembrain_ffi.so`
- macOS: `target/debug/libmembrain_ffi.dylib`
- Windows: `target/debug/membrain_ffi.dll`

Release builds are reserved for benchmarking and reproducing
performance issues — ordinary development and CI use debug.

### 3. Run Tests

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p membrain-core

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_store_fact
```

### 4. Test Language Bindings

**Python:**
```bash
cd membrain-py
pip install -e ".[dev]"           # installs ruff, mypy, pytest-asyncio
export MEMBRAIN_LIB_PATH=../target/debug/libmembrain_ffi.so
pytest
```

**JavaScript:**
```bash
cd membrain-node
export MEMBRAIN_LIB_PATH=../target/debug/libmembrain_ffi.so
npm install
npm run typecheck
npm run lint
npm test
```

## Project Layout

```
.github/workflows/       # CI: rust.yml, python.yml, node.yml, security.yml
crates/
  membrain-core/         # Core types, traits, config, errors
  membrain-storage/      # Persistence backends
  membrain-index/        # Logical index adapter over memscaledb
  membrain-pipeline/     # Write + retrieval pipelines
  membrain-compression/  # Memory consolidation, decay
  membrain-jobs/         # Background job scheduler
  membrain-audit/        # Audit log + metrics
  membrain-multi-agent/  # Multi-agent trust/visibility/sharing
  membrain-graph/        # Neural graph memory layer
  membrain-conflict/     # Conflict resolution (OpenAI-backed)
  membrain-ffi/          # C ABI cdylib
    src/c_api/           # Per-feature FFI modules
    src/c_api/safety.rs  # Bounds-checked slice helpers (safe_slice)
  memscaledb/             # Vector DB: HNSW/IVF/LSH/Vamana + persistence
membrain-py/             # Python ctypes wrapper (Conversation + AsyncConversation)
membrain-node/           # Node.js koffi wrapper
benchmarks/              # LongMemEval / LoCoMo / BEAM
docs/                    # Mintlify docs + cookbooks
examples/                # Runnable examples (Python + JS)
tests/python|node/       # Cross-language cookbook integration tests
paper/                   # Research paper source (main.tex)
```

## Development Workflow

### Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Keep changes focused** - One feature or fix per PR

3. **Write tests** for new functionality

4. **Run checks** before committing:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --lib -- -D warnings   # strict: lib-only
   cargo clippy --workspace --all-targets          # permissive: full
   cargo test --workspace
   ```
   Optional: install pre-commit hooks:
   ```bash
   pip install pre-commit && pre-commit install
   ```
   Hooks run fmt/ruff/tsc/gitleaks on every commit.

5. **Commit with clear messages** (see below)

6. **Push and create a pull request**:
   ```bash
   git push origin feature/my-feature
   ```

### Code Style

- `cargo fmt` is enforced in CI (see `.github/workflows/rust.yml`).
- `[workspace.lints.clippy]` in root `Cargo.toml` denies
  `unwrap_used`, `expect_used`, `panic`, `unreachable`, `todo`,
  `unimplemented`, `dbg_macro`. Test modules allow these via
  `#![cfg_attr(test, allow(...))]` in each crate's `lib.rs`.
- `unsafe_op_in_unsafe_fn` is `deny` — every intrinsic call inside an
  `unsafe fn` needs an explicit `unsafe { }` block with a `// SAFETY:`
  comment.
- Follow Rust naming conventions (snake_case for functions, CamelCase
  for types).
- Prefer `if let` / `match` over `.unwrap()` / `.expect()` in production
  code. Use `?` for fallible chains.
- Keep functions small and focused.
- Add `///` doc comments for public APIs.
- Update `docs/` when changing behavior.

## Adding New Features

### Adding a New Memory Type

Order per project policy (Python first, then Node, then docs):

1. Define the type in `crates/membrain-core/src/memory/`.
2. Add to the `MemoryType` enum + implement serialization.
3. Add storage support in `crates/membrain-storage/`.
4. Add an FFI function (see next section).
5. Add Python binding in `membrain-py/membrain/client.py` (+ async
   variant if applicable).
6. Add JavaScript binding in `membrain-node/src/client.ts`.
7. Add Mintlify docs under `docs/`.
8. Write unit tests (Rust) + integration tests (`tests/python/`,
   `tests/node/`).

### Adding an FFI Function

When adding a new C API function:

1. **Rust implementation** — add method to `MembrainClient` in
   `crates/membrain-ffi/src/client.rs`.
2. **C export** — add `extern "C"` wrapper in the appropriate module
   under `crates/membrain-ffi/src/c_api/` (e.g. `client.rs`, `hnsw.rs`,
   `graph.rs`). If the function takes a raw pointer + length, route it
   through `safe_slice` from `c_api/safety.rs` so null / alignment /
   overflow are checked uniformly.
3. **Python binding** — update `membrain-py/membrain/client.py`:
   add function signature in `_setup_signatures()`, then the Python
   method. Use `_parse_ffi_json` / `_read_ffi_string` for return
   payloads.
4. **Node binding** — update `membrain-node/src/client.ts`:
   add a typed `FFIFn` field and bind it in `bindFunctions()`.
5. **Docs** — update `docs/python-api.md`, `docs/javascript-api.md`,
   and add a cookbook entry.

`unsafe extern "C" fn` bodies must wrap intrinsic / raw-pointer ops
in explicit `unsafe { }` blocks with `// SAFETY:` comments.

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_fact() {
        let client = MembrainClient::new();
        let result = client.store_fact("Test fact", 0.9);
        assert!(result.success);
        assert!(result.id.is_some());
    }

    #[test]
    fn test_search() {
        let client = MembrainClient::new();
        client.store_fact("Rust is fast", 0.9);
        let results = client.search("Rust", 5);
        assert!(!results.memories.is_empty());
    }
}
```

## Commit Messages

Use clear, imperative-style commit messages:

**Good:**
```
Add store_entity FFI function for entity memory
Fix confidence clamping in preference storage
Update search to use cosine similarity
```

**Avoid:**
```
Fixed stuff
WIP
Updated files
```

Format (Conventional Commits):
```
<type>(<scope>): <short summary>

<optional detailed description>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`,
`ci`, `build`, `revert`. Scope is the crate or package name
(`ffi`, `pipeline`, `memscaledb`, `py`, `node`, etc.).

Git history was squashed once at project import. From that point on,
every change is a single logical commit — no further mass imports.
Direct force-pushes to `main` are blocked.

## Pull Request Process

1. **Update documentation** if you changed behavior
2. **Add tests** for new features
3. **Run all checks** (CI runs these on every PR):
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --lib -- -D warnings
   cargo test --workspace
   cd membrain-py && pytest && ruff check && mypy membrain
   cd membrain-node && npm run typecheck && npm run lint && npm test
   ```
4. **Update CHANGELOG.md** if applicable
5. **Fill out PR template** with description of changes
6. **Link related issues** using "Fixes #123" or "Closes #456"
7. **Be responsive** to review feedback

### PR Checklist

- [ ] Code follows project style
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No unnecessary dependencies added
- [ ] FFI changes include Python and JS bindings
- [ ] Examples/cookbooks added for new features

## Documentation

- Update API docs in `docs/python-api.md` and `docs/javascript-api.md`
- Add practical examples to `docs/cookbooks/`
- Include code comments for complex logic
- Update `README.md` if adding major features

## Bug Reports

When filing a bug report, include:

1. **Version** - Membrain version and platform
2. **Description** - What happened vs what you expected
3. **Reproduction** - Minimal code to reproduce the issue
4. **Environment** - OS, Rust version, Python/Node version
5. **Logs/Errors** - Any error messages or stack traces

## Feature Requests

For feature requests:

1. **Use Case** - Describe the problem you're trying to solve
2. **Proposed Solution** - How you envision it working
3. **Alternatives** - Other approaches you've considered
4. **Examples** - Code examples if applicable

## Performance Considerations

- Benchmark changes that affect hot paths
- Use `cargo bench` for microbenchmarks
- Profile before optimizing
- Document performance characteristics

## Security

- Report security issues privately to maintainers.
- Secrets (API keys, tokens) belong in `.env` (gitignored); the committed
  template is `.env.example`. Pre-commit runs `gitleaks`, and
  `.github/workflows/security.yml` runs `cargo audit`, `pip-audit`,
  `npm audit`, and `gitleaks` on every PR.
- FFI boundary: every raw pointer + length pair **must** be validated
  through `safe_slice` (null / alignment / length overflow) before use.
- `unsafe` blocks require a `// SAFETY:` comment and must not grow
  without review.
- Prefer safe patterns (`if let`, `?`, `match`) over `.unwrap()` /
  `.expect()` / `panic!` — these are workspace-level deny lints.

## Community

- Be respectful and constructive
- Help others in discussions and issues
- Share your use cases and feedback
- Contribute to documentation and examples

## Questions?

- Open an issue for questions
- Check existing documentation in `docs/`
- Review cookbooks for examples

Thank you for contributing to Membrain!
