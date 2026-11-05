"""Type definitions and TypedDict structures for indexing."""

from typing import TypedDict

from ..models import MinimalSource


class ChunkRecord(TypedDict):
    """Typed dictionary representing a chunk source and its text.

    Attributes:
        source: MinimalSource metadata identifying chunk bounds.
        text: Plain text content slice of the chunk.
    """

    source: MinimalSource
    text: str


class ManifestRecord(TypedDict):
    """Typed dictionary representing a file hash and mtime.

    Attributes:
        hash: SHA-256 hexadecimal content hash string.
        mtime: File modification timestamp float.
    """

    hash: str
    mtime: float
