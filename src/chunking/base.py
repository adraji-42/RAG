"""Base class and common types for syntax-aware document chunking."""

from typing import List, Tuple
from abc import ABC, abstractmethod

from ..models import MinimalSource

Span = Tuple[int, int]


class BaseChunker(ABC):
    """Abstract base class for syntax-aware document chunkers."""

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        """Initialize the base chunker with a maximum character limit.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
        """
        from .emitter import Emitter

        self.__max_chunk_size: int = max_chunk_size
        self.__emitter: Emitter = Emitter(max_chunk_size)

    @property
    def max_chunk_size(self) -> int:
        """Return the maximum allowed chunk size in characters.

        Returns:
            Maximum allowed chunk size integer in characters.
        """
        return self.__max_chunk_size

    def emit(
        self, fp: str, content: str, spans: List[Span],
    ) -> List[MinimalSource]:
        """Emit packed and split sources using the internal emitter.

        Args:
            fp: Source document file path string.
            content: Full text content of the document.
            spans: List of character (start, end) tuples.

        Returns:
            List of MinimalSource objects within size constraints.
        """
        return self.__emitter.emit(fp, content, spans)

    @abstractmethod
    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        """Chunk document content into character-indexed minimal sources.

        Args:
            file_path: Source document file path string.
            content: Full text content of the document.

        Returns:
            List of MinimalSource chunks extracted from content.
        """
        ...
