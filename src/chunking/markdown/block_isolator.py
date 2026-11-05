"""Protected code and table block isolation for markdown chunking."""

from typing import List, Tuple

from ..base import Span
from ..text import TextChunker


class BlockIsolator:
    """Isolate code and table blocks from regular markdown prose."""

    def __init__(
        self, text_chunker: TextChunker,
        max_chunk_size: int,
    ) -> None:
        """Initialize the block isolator with text chunker and size limit.

        Args:
            text_chunker: TextChunker instance for splitting prose.
            max_chunk_size: Maximum allowed characters per chunk.
        """
        self.__text: TextChunker = text_chunker
        self.__max: int = max_chunk_size

    def isolate(
        self, content: str, start: int, end: int,
        blocks: List[Tuple[str, Span]],
    ) -> List[Span]:
        """Separate text range into protected blocks and plain text spans.

        Args:
            content: Markdown document text string.
            start: Region start character offset.
            end: Region end character offset.
            blocks: List of protected code/table block tuples.

        Returns:
            List of character span tuples for isolated blocks and prose.
        """
        relevant: List[Tuple[str, Span]] = [
            (k, (s, e)) for k, (s, e) in blocks
            if s >= start and e <= end
        ]
        if not relevant:
            return self.__text.split_block(
                content, start, end,
            )
        return self.__dispatch(
            content, start, end, relevant,
        )

    def __dispatch(
        self, content: str, start: int, end: int,
        relevant: List[Tuple[str, Span]],
    ) -> List[Span]:
        """Process segments between protected blocks using text splitters.

        Args:
            content: Markdown document text string.
            start: Region start character offset.
            end: Region end character offset.
            relevant: List of relevant block tuples within the range.

        Returns:
            List of character span tuples dispatched across boundaries.
        """
        spans: List[Span] = []
        pos: int = start
        for _, (bs, be) in relevant:
            if bs < pos:
                continue
            elif bs > pos:
                spans.extend(self.__text.split_block(
                    content, pos, bs,
                ))
            if be - bs <= self.__max:
                spans.append((bs, be))
            else:
                spans.extend(self.__text.split_lines(
                    content, bs, be,
                ))
            pos = be
        if pos < end:
            spans.extend(self.__text.split_block(
                content, pos, end,
            ))
        return spans
