"""Public API for the file chunking subsystem."""

from __future__ import annotations

import os
from typing import Iterator

from .base import BaseChunker, Chunk
from .markdown import MarkdownChunker
from .python import PythonChunker

__all__ = ["BaseChunker", "Chunk", "ChunkerFactory", "ChunkingOrchestrator"]

_PYTHON_EXTENSIONS: frozenset[str] = frozenset({".py", ".pyi"})


class ChunkerFactory:
    """Selects the correct :class:`BaseChunker` for a file extension."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialise with a shared *max_chunk_size* for every chunker."""
        self.max_chunk_size = max_chunk_size

    def get_chunker(self, file_path: str) -> BaseChunker:
        """Return the :class:`BaseChunker` matching *file_path*'s extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _PYTHON_EXTENSIONS:
            return PythonChunker(self.max_chunk_size)
        return MarkdownChunker(self.max_chunk_size)


class ChunkingOrchestrator:
    """Coordinates chunking of files via a :class:`ChunkerFactory`."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialise the orchestrator with a configurable *max_chunk_size*."""
        self._factory = ChunkerFactory(max_chunk_size)

    def process(self, file_path: str) -> Iterator[Chunk]:
        """Dispatch *file_path* to the correct strategy and yield all chunks.

        Yields :class:`Chunk` instances for every segment produced.
        Unknown extensions fall back to :class:`MarkdownChunker`.
        """
        chunker = self._factory.get_chunker(file_path)
        yield from chunker.chunk_file(file_path)
