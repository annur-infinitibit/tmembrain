"""
Pytest configuration and fixtures for Membrain tests
"""
import os
import tempfile
import shutil
import pytest


@pytest.fixture
def temp_storage(tmp_path):
    """Provide a temporary storage directory for tests"""
    storage_dir = tmp_path / "membrain_test_storage"
    storage_dir.mkdir()
    yield str(storage_dir)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def membrain_config(temp_storage):
    """Provide a test configuration with temporary storage"""
    return {
        "storage_path": temp_storage,
        "max_memories": 100000,
        "embedding_dim": 16,  # Small for testing
        "similarity_threshold": 0.70,  # Lower threshold for testing
    }


@pytest.fixture(autouse=True)
def set_lib_path():
    """Automatically set library path for all tests"""
    if 'MEMBRAIN_LIB_PATH' not in os.environ:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Prefer debug build, fall back to release
        debug_path = os.path.join(project_root, 'target', 'debug', 'libmembrain_ffi.so')
        release_path = os.path.join(project_root, 'target', 'release', 'libmembrain_ffi.so')
        if os.path.exists(debug_path):
            os.environ['MEMBRAIN_LIB_PATH'] = debug_path
        elif os.path.exists(release_path):
            os.environ['MEMBRAIN_LIB_PATH'] = release_path


@pytest.fixture(autouse=True)
def isolate_default_storage(tmp_path, monkeypatch):
    """Point the default MembrainClient() storage at a unique per-test directory.

    Tests that call `MembrainClient()` with no config rely on the Rust FFI's
    `Config::from_env()` path, which honours MEMBRAIN_STORAGE_PATH. Without this
    isolation, every `MembrainClient()` would share ./memscaledb in the working
    directory, causing cross-test contamination.
    """
    import uuid
    storage_path = tmp_path / f"default_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("MEMBRAIN_STORAGE_PATH", str(storage_path))


@pytest.fixture
def unique_storage_config(tmp_path):
    """Provide a unique storage configuration for each test"""
    import uuid
    storage_path = tmp_path / f"test_{uuid.uuid4().hex[:8]}"
    return {
        "storage": {
            "backend": "memscaledb",
            "path": str(storage_path)
        }
    }


@pytest.fixture(scope="session")
def openai_available():
    """Validate OPENAI_API_KEY once per session; skip tests if missing/invalid.

    Tests that make live OpenAI calls should depend on this fixture so they
    skip cleanly on auth failure instead of flooding the report with 401s.
    """
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set", allow_module_level=False)
    try:
        from openai import OpenAI, AuthenticationError
        client = OpenAI(api_key=key)
        client.models.list()
    except AuthenticationError as error:
        pytest.skip(f"OPENAI_API_KEY invalid: {error}")
    except Exception as error:
        pytest.skip(f"OpenAI unreachable: {error}")
    return True
