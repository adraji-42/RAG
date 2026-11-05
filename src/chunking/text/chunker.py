"""Plain text chunker using recursive boundary splitting."""

from typing import List

from ...models import MinimalSource
from ..base import BaseChunker, Span
from .splitter import RecursiveTextSplitter


class TextChunker(BaseChunker):
    """Chunk arbitrary plain text using recursive boundary splitting."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the text chunker with a recursive text splitter.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
        """
        super().__init__(max_chunk_size)
        self.__splitter: RecursiveTextSplitter = RecursiveTextSplitter(
            max_chunk_size,
        )

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        """Split text into recursive spans and emit minimal source chunks.

        Args:
            file_path: Source plain text file path string.
            content: Plain text file content.

        Returns:
            List of MinimalSource chunks extracted recursively.
        """
        spans: List[Span] = self.__splitter.split_block(
            content, 0, len(content)
        )
        return self.emit(file_path, content, spans)

    def split_block(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        """Delegate block splitting to internal recursive text splitter.

        Args:
            content: Document text string.
            start: Start character index in content.
            end: End character index in content.

        Returns:
            List of character span tuples for the block.
        """
        return self.__splitter.split_block(content, start, end)

    def split_lines(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        """Delegate line splitting to internal recursive text splitter.

        Args:
            content: Document text string.
            start: Start character index in content.
            end: End character index in content.

        Returns:
            List of character span tuples split along lines.
        """
        return self.__splitter.split_lines(content, start, end)
