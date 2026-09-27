"""Structural Markdown / plain-text file chunker implementing BaseChunker."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .base import BaseChunker, Chunk


_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)


class MarkdownChunker(BaseChunker):
    """Chunks Markdown and plain-text files by structural semantic blocks."""

    def chunk(self, file_path: Path, content: str) -> Iterator[Chunk]:
        """Yield chunks for *content* split at Markdown structural bounds."""
        points = self._find_split_points(content)
        segments = self._segments_from_points(content, points, len(content))
        for seg_start, segment in segments:
            yield from self._emit_segment(
                content, seg_start, segment, str(file_path),
            )

    @staticmethod
    def _fenced_regions(text: str) -> list[tuple[int, int]]:
        """Return ``(start, end)`` spans covered by fenced code blocks.

        Handles ``` and ~~~ fences, including ones indented under a
        list item. An unterminated fence is treated as covering the
        rest of the file, so nothing after it is misread as Markdown
        structure.
        """
        regions: list[tuple[int, int]] = []
        open_start: int | None = None
        for match in _FENCE_RE.finditer(text):
            if open_start is None:
                open_start = match.start()
            else:
                regions.append((open_start, match.end()))
                open_start = None
        if open_start is not None:
            regions.append((open_start, len(text)))
        return regions

    @staticmethod
    def _in_fenced_region(pos: int, regions: list[tuple[int, int]]) -> bool:
        """Return whether char offset *pos* falls inside a fenced region."""
        return any(start <= pos < end for start, end in regions)

    @classmethod
    def _find_split_points(cls, text: str) -> list[int]:
        """Return sorted char offsets where a new structural block begins.

        A header only counts as a split point when it sits outside a
        fenced code block: an inline ``#`` comment inside an example
        snippet must not be read as a Markdown heading.
        """
        fenced = cls._fenced_regions(text)
        points: set[int] = {0}
        for match in _HEADER_RE.finditer(text):
            if not cls._in_fenced_region(match.start(), fenced):
                points.add(match.start())
        for start, _ in fenced:
            points.add(start)
        return sorted(points)

    @staticmethod
    def _paragraph_split_points(text: str, base: int) -> list[int]:
        """Return char offsets of paragraph boundaries within *text*."""
        points: set[int] = {base}
        for m in re.finditer(r"\n{2,}", text):
            points.add(base + m.end())
        return sorted(points)

    @staticmethod
    def _segments_from_points(
        text: str,
        points: list[int],
        region_end: int,
    ) -> Iterator[tuple[int, str]]:
        """Yield ``(start, segment_text)`` pairs between split *points*.

        *region_end* bounds the final segment so callers can reuse this
        both for the whole document and for a single oversized segment
        being split further. Whitespace-only segments are dropped: they
        carry no retrievable content and would only waste an index slot.
        """
        for i, start in enumerate(points):
            end = points[i + 1] if i + 1 < len(points) else region_end
            segment = text[start:end]
            if segment.strip():
                yield start, segment

    def _emit_segment(
        self,
        source: str,
        seg_start: int,
        segment: str,
        file_path: str,
    ) -> Iterator[Chunk]:
        """Yield chunks for one structural segment, splitting as needed."""
        if len(segment) <= self.max_chunk_size:
            yield Chunk.spanning(file_path, seg_start, segment)
            return
        seg_end = seg_start + len(segment)
        para_points = self._paragraph_split_points(segment, seg_start)
        paragraphs = self._segments_from_points(source, para_points, seg_end)
        for para_start, para in paragraphs:
            yield from self._emit_paragraph(para, para_start, file_path)

    def _emit_paragraph(
        self,
        para: str,
        para_start: int,
        file_path: str,
    ) -> Iterator[Chunk]:
        """Yield one or more chunks for a single paragraph *para*."""
        if len(para) <= self.max_chunk_size:
            yield Chunk.spanning(file_path, para_start, para)
        else:
            yield from self._split_by_size(para, para_start, file_path)
