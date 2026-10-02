import re
from typing import List, Tuple

from .base import BaseChunker, Span
from .line_map import LineMap
from ..models import MinimalSource

_FENCE_RE: re.Pattern[str] = re.compile(r"^(`{3,}|~{3,})")
_ATX_RE: re.Pattern[str] = re.compile(r"^(#{1,6})(?:\s|$)")
_SETEXT_H1: re.Pattern[str] = re.compile(r"^=+\s*$")
_SETEXT_H2: re.Pattern[str] = re.compile(r"^-+\s*$")


class FenceScanner:

    def __init__(self, lines: List[str]) -> None:
        self._lines: List[str] = lines
        self.mask: List[bool] = []
        in_f: bool = False
        mk: str = ""
        for line in lines:
            m: re.Match[str] | None = _FENCE_RE.match(
                line.strip(),
            )
            if m is not None:
                tok: str = m.group(1)
                if not in_f:
                    in_f, mk = True, tok[0]
                elif tok[0] == mk and len(tok) >= 3:
                    in_f, mk = False, ""
            self.mask.append(in_f)

    def boundaries(
        self, lmap: LineMap,
        first: int, last: int, span_start: int,
    ) -> List[int]:
        bounds: List[int] = []
        inside: bool = False
        for idx in range(first, last + 1):
            off: int = lmap.line_start(idx)
            is_fence: bool = bool(
                _FENCE_RE.match(self._lines[idx].strip()),
            )
            if off <= span_start:
                inside = not inside if is_fence else inside
                continue
            if not is_fence:
                continue
            if not inside:
                bounds.append(lmap.line_start(idx))
                inside = True
            else:
                bounds.append(lmap.line_start(idx + 1))
                inside = False
        return bounds


class HeadingScanner:

    def __init__(
        self, lines: List[str],
        fence_mask: List[bool],
    ) -> None:
        self._lines: List[str] = lines
        self._mask: List[bool] = fence_mask

    def level(self, idx: int) -> int:
        if self._mask[idx]:
            return 0
        atx: re.Match[str] | None = _ATX_RE.match(
            self._lines[idx],
        )
        if atx:
            return len(atx.group(1))
        if idx + 1 >= len(self._lines):
            return 0
        if not self._lines[idx].strip():
            return 0
        nxt: str = self._lines[idx + 1]
        if _SETEXT_H1.match(nxt):
            return 1
        return 2 if _SETEXT_H2.match(nxt) else 0

    def find(
        self, lmap: LineMap,
        start: int, end: int, plvl: int,
    ) -> List[Tuple[int, int]]:
        first: int = lmap.line_at(start)
        last: int = lmap.line_at(end - 1)
        cands: List[Tuple[int, int]] = []
        for i in range(first, last + 1):
            if lmap.line_start(i) <= start:
                continue
            lv: int = self.level(i)
            if lv > plvl:
                cands.append((lv, i))
        if not cands:
            return []
        bv: int = min(v for v, _ in cands)
        return [(v, i) for v, i in cands if v == bv]


class MarkdownChunker(BaseChunker):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        super().__init__(max_chunk_size)
        self._content: str = ""
        self._lmap: LineMap = LineMap("")
        self._fences: FenceScanner = FenceScanner([])
        self._heads: HeadingScanner = (
            HeadingScanner([], [])
        )

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        self._content = content
        lines: List[str] = content.split("\n")
        self._lmap = LineMap(content)
        self._fences = FenceScanner(lines)
        self._heads = HeadingScanner(
            lines, self._fences.mask,
        )
        spans: List[Span] = self._split(
            0, len(content), 0,
        )
        return self._emit(file_path, content, spans)

    def _split(
        self, start: int, end: int, plvl: int,
    ) -> List[Span]:
        if end - start <= self._max_chunk_size:
            return [(start, end)]
        hdgs: List[Tuple[int, int]] = (
            self._heads.find(
                self._lmap, start, end, plvl,
            )
        )
        if hdgs:
            return self._at_hdgs(start, end, hdgs)
        return self._fallback(start, end)

    def _at_hdgs(
        self, start: int, end: int,
        hdgs: List[Tuple[int, int]],
    ) -> List[Span]:
        lv: int = hdgs[0][0]
        pts: List[int] = [
            self._lmap.line_start(i) for _, i in hdgs
        ]
        spans: List[Span] = []
        prev: int = start
        for p in pts:
            if p > prev:
                spans.extend(self._split(prev, p, lv))
            prev = p
        if prev < end:
            spans.extend(self._split(prev, end, lv))
        return spans

    def _fallback(
        self, start: int, end: int,
    ) -> List[Span]:
        fb: List[int] = self._fences.boundaries(
            self._lmap, self._lmap.line_at(start),
            self._lmap.line_at(end - 1), start,
        )
        if fb:
            fs: List[Span] = self._bounds_to_spans(
                fb, start, end,
            )
            if self._is_useful(fs, start, end):
                return self._recurse_fallback(fs)
        ps: List[Span] = self._split_paragraphs(
            self._content, start, end,
        )
        if self._is_useful(ps, start, end):
            return self._recurse_fallback(ps)
        return [(start, end)]

    def _is_useful(
        self, spans: List[Span],
        start: int, end: int,
    ) -> bool:
        if len(spans) <= 1:
            return False
        return max(e - s for s, e in spans) < end - start

    def _recurse_fallback(
        self, spans: List[Span],
    ) -> List[Span]:
        result: List[Span] = []
        for s, e in spans:
            if e - s <= self._max_chunk_size:
                result.append((s, e))
            else:
                result.extend(self._fallback(s, e))
        return result

    def _bounds_to_spans(
        self, bounds: List[int],
        start: int, end: int,
    ) -> List[Span]:
        spans: List[Span] = []
        prev: int = start
        for b in bounds:
            if b > prev:
                spans.append((prev, b))
            prev = b
        if prev < end:
            spans.append((prev, end))
        return spans
