"""Membrain error types."""


class MembrainError(Exception):
    """Base exception for Membrain operations."""

    def __init__(self, message: str, code: int = -1) -> None:
        super().__init__(message)
        self.code = code
