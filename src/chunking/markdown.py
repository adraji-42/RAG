import re

from typing import List, Tuple

from .base import BaseChunker
from ..models import MinimalSource

_FENCE_RE: re.Pattern[str] = re.compile(r"^(`{3,}|~{3,})")
_ATX_RE: re.Pattern[str] = re.compile(r"^(#{1,6})(?:\s|$)")
_SETEXT_H1_RE: re.Pattern[str] = re.compile(r"^=+\s*$")
_SETEXT_H2_RE: re.Pattern[str] = re.compile(r"^-+\s*$")


class MarkdownChunker(BaseChunker):

    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)
        self._lines: List[str] = []
        self._line_offsets: List[int] = []
        self._fence_mask: List[bool] = []
        self._content: str = ""

    def chunk(
        self,
        file_path: str,
        content: str,
    ) -> List[MinimalSource]:
        self._content = content
        self._lines = content.split("\n")
        self._line_offsets = self._build_line_offsets(
            content,
        )
        self._fence_mask = self._build_fence_mask()
        spans: List[Tuple[int, int]] = (
            self._hierarchical_split(0, len(content), 0)
        )
        return self._spans_to_chunks(
            file_path, content, spans,
        )

    def _build_fence_mask(self) -> List[bool]:
        mask: List[bool] = []
        in_fence: bool = False
        marker: str = ""
        for line in self._lines:
            in_fence, marker = self._update_fence(
                line, in_fence, marker,
            )
            mask.append(in_fence)
        return mask

    def _update_fence(
        self,
        line: str,
        in_fence: bool,
        fence_marker: str,
    ) -> Tuple[bool, str]:
        match: re.Match[str] | None = _FENCE_RE.match(
            line.strip(),
        )
        if match is None:
            return in_fence, fence_marker
        marker: str = match.group(1)
        return self._toggle_fence(
            in_fence, fence_marker, marker,
        )

    def _toggle_fence(
        self,
        in_fence: bool,
        fence_marker: str,
        marker: str,
    ) -> Tuple[bool, str]:
        if not in_fence:
            return True, marker[0]
        if (
            marker[0] == fence_marker
            and len(marker) >= 3
        ):
            return False, ""
        return True, fence_marker

    def _heading_level(self, idx: int) -> int:
        if self._fence_mask[idx]:
            return 0
        line: str = self._lines[idx]
        atx: re.Match[str] | None = _ATX_RE.match(line)
        if atx:
            return len(atx.group(1))
        return self._setext_level(idx)

    def _setext_level(self, idx: int) -> int:
        if idx + 1 >= len(self._lines):
            return 0
        current: str = self._lines[idx].strip()
        if not current:
            return 0
        nxt: str = self._lines[idx + 1]
        if _SETEXT_H1_RE.match(nxt):
            return 1
        if _SETEXT_H2_RE.match(nxt):
            return 2
        return 0

    def _line_for_offset(self, offset: int) -> int:
        lo: int = 0
        hi: int = len(self._line_offsets) - 1
        while lo <= hi:
            mid: int = (lo + hi) // 2
            if self._line_offsets[mid] <= offset:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    def _line_offset(self, line_idx: int) -> int:
        if line_idx < len(self._line_offsets):
            return self._line_offsets[line_idx]
        return len(self._content)

    def _hierarchical_split(
        self,
        start: int,
        end: int,
        parent_level: int,
    ) -> List[Tuple[int, int]]:
        if end - start <= self._max_chunk_size:
            return [(start, end)]
        headings: List[Tuple[int, int]] = (
            self._find_headings_in_range(
                start, end, parent_level,
            )
        )
        if headings:
            return self._split_by_headings(
                start, end, headings,
            )
        return self._headerless_fallback(start, end)

    def _find_headings_in_range(
        self,
        start: int,
        end: int,
        parent_level: int,
    ) -> List[Tuple[int, int]]:
        first_line: int = self._line_for_offset(start)
        last_line: int = self._line_for_offset(end - 1)
        candidates: List[Tuple[int, int]] = []
        for idx in range(first_line, last_line + 1):
            off: int = self._line_offset(idx)
            if off <= start:
                continue
            lvl: int = self._heading_level(idx)
            if lvl > parent_level:
                candidates.append((lvl, idx))
        return self._select_best_headings(candidates)

    def _select_best_headings(
        self,
        candidates: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not candidates:
            return []
        best: int = min(lvl for lvl, _ in candidates)
        return [
            (lvl, idx)
            for lvl, idx in candidates
            if lvl == best
        ]

    def _split_by_headings(
        self,
        start: int,
        end: int,
        headings: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        level: int = headings[0][0]
        positions: List[int] = [
            self._line_offset(idx)
            for _, idx in headings
        ]
        spans: List[Tuple[int, int]] = []
        prev: int = start
        for pos in positions:
            if pos > prev:
                spans.extend(
                    self._hierarchical_split(
                        prev, pos, level,
                    )
                )
            prev = pos
        if prev < end:
            spans.extend(
                self._hierarchical_split(
                    prev, end, level,
                )
            )
        return spans

    def _headerless_fallback(
        self,
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        fence_spans: List[Tuple[int, int]] = (
            self._split_by_fences(start, end)
        )
        if self._made_progress(fence_spans, start, end):
            return self._recurse_spans(fence_spans)
        para_spans: List[Tuple[int, int]] = (
            self._split_by_paragraphs(start, end)
        )
        if self._made_progress(para_spans, start, end):
            return self._recurse_spans(para_spans)
        return [(start, end)]

    def _made_progress(
        self,
        spans: List[Tuple[int, int]],
        start: int,
        end: int,
    ) -> bool:
        if len(spans) <= 1:
            return False
        original: int = end - start
        largest: int = max(e - s for s, e in spans)
        return largest < original

    def _recurse_spans(
        self,
        spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for s, e in spans:
            if e - s <= self._max_chunk_size:
                result.append((s, e))
            else:
                result.extend(
                    self._headerless_fallback(s, e),
                )
        return result

    def _split_by_fences(
        self,
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        first: int = self._line_for_offset(start)
        last: int = self._line_for_offset(end - 1)
        boundaries: List[int] = self._fence_boundaries(
            first, last, start,
        )
        if not boundaries:
            return [(start, end)]
        return self._offsets_to_spans(
            boundaries, start, end,
        )

    def _fence_boundaries(
        self,
        first: int,
        last: int,
        span_start: int,
    ) -> List[int]:
        boundaries: List[int] = []
        in_fence: bool = False
        for idx in range(first, last + 1):
            off: int = self._line_offset(idx)
            if off <= span_start:
                stripped = self._lines[idx].strip()
                if _FENCE_RE.match(stripped):
                    in_fence = not in_fence
                continue
            stripped = self._lines[idx].strip()
            if not _FENCE_RE.match(stripped):
                continue
            if not in_fence:
                boundaries.append(off)
                in_fence = True
            else:
                boundaries.append(
                    self._line_offset(idx + 1),
                )
                in_fence = False
        return boundaries

    def _split_by_paragraphs(
        self,
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        text: str = self._content[start:end]
        parts: List[str] = text.split("\n\n")
        if len(parts) <= 1:
            return [(start, end)]
        spans: List[Tuple[int, int]] = []
        pos: int = start
        for part in parts:
            seg_end: int = pos + len(part)
            if seg_end < end:
                seg_end += 2
            seg_end = min(seg_end, end)
            spans.append((pos, seg_end))
            pos = seg_end
        return spans

    def _offsets_to_spans(
        self,
        boundaries: List[int],
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        prev: int = start
        for b in boundaries:
            if b > prev:
                spans.append((prev, b))
            prev = b
        if prev < end:
            spans.append((prev, end))
        return spans
