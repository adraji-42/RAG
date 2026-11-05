"""Hierarchical heading-based text splitter for markdown documents."""

from typing import List, Tuple

from ..base import Span
from .block_isolator import BlockIsolator


class HeadingSplitter:
    """Split markdown content hierarchically by heading levels."""

    def __init__(
        self, isolator: BlockIsolator,
        max_chunk_size: int,
    ) -> None:
        """Initialize heading splitter with block isolator and max size.

        Args:
            isolator: BlockIsolator instance for processing block regions.
            max_chunk_size: Maximum allowed characters per chunk.
        """
        self.__isolator: BlockIsolator = isolator
        self.__max: int = max_chunk_size

    def split(
        self, content: str,
        start: int, end: int, plvl: int,
        heads: List[Tuple[int, int]],
        blocks: List[Tuple[str, Span]],
    ) -> List[Span]:
        """Recursively split markdown section at subheadings or fallback.

        Args:
            content: Markdown document text string.
            start: Region start character offset.
            end: Region end character offset.
            plvl: Parent heading level integer.
            heads: List of (heading_level, character_offset) tuples.
            blocks: List of protected code/table block tuples.

        Returns:
            List of character span tuples split at subheadings.
        """
        if end - start <= self.__max:
            return [(start, end)]
        pts: List[int] = self.__heading_pts(heads, start, end, plvl)
        if pts:
            return self.__split_at(content, start, end, pts, heads, blocks)
        return self.__isolator.isolate(content, start, end, blocks)

    def __heading_pts(
        self, heads: List[Tuple[int, int]],
        start: int, end: int, plvl: int,
    ) -> List[int]:
        """Find offsets of candidate subheadings at shallowest level.

        Args:
            heads: List of (heading_level, character_offset) tuples.
            start: Region start character offset.
            end: Region end character offset.
            plvl: Parent heading level integer.

        Returns:
            List of character offsets for shallowest subheadings.
        """
        cands: List[Tuple[int, int]] = [
            (lv, off) for lv, off in heads
            if start < off < end and lv > plvl
        ]
        if not cands:
            return []
        best: int = min(lv for lv, _ in cands)
        return sorted(off for lv, off in cands if lv == best)

    def __split_at(
        self, content: str, start: int, end: int, pts: List[int],
        heads: List[Tuple[int, int]], blocks: List[Tuple[str, Span]]
    ) -> List[Span]:
        """Split range at heading offsets and recurse on sections.

        Args:
            content: Markdown document text string.
            start: Region start character offset.
            end: Region end character offset.
            pts: List of heading character offsets to split on.
            heads: List of (heading_level, character_offset) tuples.
            blocks: List of protected block tuples.

        Returns:
            List of character span tuples for split sub-sections.
        """
        prev: int = start
        spans: List[Span] = []
        best: int = min(lv for lv, off in heads if off in pts)
        for p in pts:
            if p > prev:
                spans.extend(self.split(
                    content, prev, p, best,
                    heads, blocks,
                ))
            prev = p
        if prev < end:
            spans.extend(self.split(
                content, prev, end, best,
                heads, blocks,
            ))
        return spans
