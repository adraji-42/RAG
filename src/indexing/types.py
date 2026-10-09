from typing import TypedDict

from ..models import MinimalSource


class ChunkRecord(TypedDict):
    source: MinimalSource
    text: str


class ManifestRecord(TypedDict):
    hash: str
    mtime: float
    chunk_count: int
