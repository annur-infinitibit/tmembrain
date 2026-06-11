"""Legacy ctypes FFI helpers — kept for reference only.

The Python client now uses the native PyO3 extension (membrain._native).
This module is not used by any active code and will be removed in v0.2.0.

For the new native extension, see::

    membrain._native  — PyO3 module (compiled Rust)
    membrain.client   — Thin async Python wrapper
"""

# All functions below are retained for backward compatibility during
# transition but are NOT called by the new client.

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from .errors import MembrainError

MEMBRAIN_OK = 0


def find_library() -> str:
    """Locate the membrain shared library (legacy ctypes path).

    .. deprecated::
        No longer used by MembrainClient. Kept for external consumers
        that may still depend on the C ABI directly.
    """
    env_path = os.environ.get("MEMBRAIN_LIB_PATH")
    if env_path:
        return env_path

    system = platform.system()
    if system == "Linux":
        lib_name = "libmembrain_ffi.so"
    elif system == "Darwin":
        lib_name = "libmembrain_ffi.dylib"
    elif system == "Windows":
        lib_name = "membrain_ffi.dll"
    else:
        lib_name = "libmembrain_ffi.so"

    pkg_dir = Path(__file__).resolve().parent
    search_dirs = [
        pkg_dir,
        pkg_dir.parent,
        pkg_dir.parent.parent / "target" / "debug",
        pkg_dir.parent.parent / "target" / "release",
    ]
    for directory in search_dirs:
        candidate = directory / lib_name
        if candidate.exists():
            return str(candidate)

    return lib_name


def encode(string: str) -> bytes:
    """Encode a Python string to null-terminated UTF-8 bytes for C."""
    return string.encode("utf-8")


def get_last_error(lib: ctypes.CDLL) -> str:
    """Retrieve the last error message from the native library."""
    err = lib.membrain_last_error()
    if err:
        return err.decode("utf-8", errors="replace")
    return "unknown error"


def check(lib: ctypes.CDLL, code: int) -> None:
    """Raise MembrainError if code is not MEMBRAIN_OK."""
    if code != MEMBRAIN_OK:
        raise MembrainError(get_last_error(lib), code=code)


def setup_error_signatures(lib: ctypes.CDLL) -> None:
    """Declare the shared error and string_free C function signatures."""
    lib.membrain_last_error.restype = ctypes.c_char_p
    lib.membrain_last_error.argtypes = []

    lib.membrain_string_free.restype = None
    lib.membrain_string_free.argtypes = [ctypes.c_char_p]


def load_library(lib_path: str | None = None) -> ctypes.CDLL:
    """Load the native library and set up error handling signatures."""
    lib_file = lib_path or find_library()
    lib = ctypes.CDLL(lib_file)
    setup_error_signatures(lib)
    return lib
