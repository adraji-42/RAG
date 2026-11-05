"""Span packing and boundary splitting emitter for text chunks."""

from typing import List

from .base import Span
from ..models import MinimalSource


class Emitter:
    """Pack small character spans and split oversized text spans."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the span emitter with a maximum chunk size.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
        """
        self.__max_chunk_size: int = max_chunk_size

    def pack(
        self, content: str, spans: List[Span],
    ) -> List[Span]:
        """Merge adjacent whitespace-separated spans within size limit.

        Args:
            content: Full document text string.
            spans: List of character (start, end) span tuples.

        Returns:
            List of merged and packed (start, end) span tuples.
        """
        clean: List[Span] = [
            (s, e) for s, e in spans
            if s < e and content[s:e].strip()
        ]
        if not clean:
            return []
        packed: List[Span] = []
        cs, ce = clean[0]
        for s, e in clean[1:]:
            if (
                content[ce:s].strip() == ""
                and e - cs <= self.__max_chunk_size
            ):
                ce = e
            else:
                packed.append((cs, ce))
                cs, ce = s, e
        packed.append((cs, ce))
        return packed

    def split_newlines(
        self, fp: str, content: str,
        start: int, end: int,
    ) -> List[MinimalSource]:
        """Split an oversized span at newline or whitespace boundaries.

        Args:
            fp: Source file path string.
            content: Full document text string.
            start: Start character index in content.
            end: End character index in content.

        Returns:
            List of MinimalSource objects split along safe boundaries.
        """
        chunks: List[MinimalSource] = []
        pos: int = start
        while pos < end:
            cut: int
            if end - pos <= self.__max_chunk_size:
                cut = end
            else:
                limit: int = pos + self.__max_chunk_size
                nl: int = content.rfind("\n", pos, limit)
                sp: int = content.rfind(" ", pos, limit)
                cut = (
                    nl + 1 if nl > pos
                    else sp + 1 if sp > pos
                    else min(limit, end)
                )
            chunks.append(MinimalSource(
                file_path=fp,
                first_character_index=pos,
                last_character_index=cut,
            ))
            pos = cut
        return chunks

    def emit(
        self, fp: str, content: str, spans: List[Span],
    ) -> List[MinimalSource]:
        """Pack spans and convert them into minimal source records.

        Args:
            fp: Source file path string.
            content: Full document text string.
            spans: List of character (start, end) span tuples.

        Returns:
            List of MinimalSource chunks within character limits.
        """
        packed: List[Span] = self.pack(content, spans)
        result: List[MinimalSource] = []
        for s, e in packed:
            if e - s <= self.__max_chunk_size:
                result.append(MinimalSource(
                    file_path=fp,
                    first_character_index=s,
                    last_character_index=e,
                ))
            else:
                result.extend(
                    self.split_newlines(fp, content, s, e),
                )
        return result
