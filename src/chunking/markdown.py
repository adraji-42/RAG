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
        self.mask: List[bool] = self._build()

    def _build(self) -> List[bool]:
        mask: List[bool] = []
        in_f: bool = False
        mk: str = ""
        for line in self._lines:
            in_f, mk = self._toggle(line, in_f, mk)
            mask.append(in_f)
        return mask

    def _toggle(
        self, line: str, in_f: bool, mk: str,
    ) -> Tuple[bool, str]:
        m: re.Match[str] | None = _FENCE_RE.match(
            line.strip(),
        )
        if m is None:
            return in_f, mk
        tok: str = m.group(1)
        if not in_f:
            return True, tok[0]
        if tok[0] == mk and len(tok) >= 3:
            return False, ""
        return True, mk

    def boundaries(
        self, lmap: LineMap,
        first: int, last: int, span_start: int,
    ) -> List[int]:
        bounds: List[int] = []
        inside: bool = False
        for idx in range(first, last + 1):
            inside = self._check(
                lmap, idx, span_start, inside, bounds,
            )
        return bounds

    def _check(
        self, lmap: LineMap, idx: int,
        span_start: int, inside: bool,
        bounds: List[int],
    ) -> bool:
        off: int = lmap.line_start(idx)
        s: str = self._lines[idx].strip()
        is_fence: bool = bool(_FENCE_RE.match(s))
        if off <= span_start:
            return not inside if is_fence else inside
        if not is_fence:
            return inside
        return self._record(lmap, idx, inside, bounds)

    def _record(
        self, lmap: LineMap, idx: int,
        inside: bool, bounds: List[int],
    ) -> bool:
        if not inside:
            bounds.append(lmap.line_start(idx))
            return True
        bounds.append(lmap.line_start(idx + 1))
        return False


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
        return self._setext(idx)

    def _setext(self, idx: int) -> int:
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
        return self._best(cands)

    def _best(
        self, cands: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
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
        self._init(content)
        spans: List[Span] = self._split(
            0, len(content), 0,
        )
        return self._packer.emit(
            file_path, content, spans,
        )

    def _init(self, content: str) -> None:
        lines: List[str] = content.split("\n")
        self._lmap = LineMap(content)
        self._fences = FenceScanner(lines)
        self._heads = HeadingScanner(
            lines, self._fences.mask,
        )

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
        return self._split_pts(start, end, pts, lv)

    def _split_pts(
        self, start: int, end: int,
        pts: List[int], lv: int,
    ) -> List[Span]:
        spans: List[Span] = []
        prev: int = start
        for p in pts:
            if p > prev:
                spans.extend(
                    self._split(prev, p, lv),
                )
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
            fs: List[Span] = self._spans(
                fb, start, end,
            )
            if self._ok(fs, start, end):
                return self._recurse(fs)
        return self._para_fb(start, end)

    def _para_fb(
        self, start: int, end: int,
    ) -> List[Span]:
        ps: List[Span] = (
            self._packer.split_paragraphs(
                self._content, start, end,
            )
        )
        if self._ok(ps, start, end):
            return self._recurse(ps)
        return [(start, end)]

    def _ok(
        self, spans: List[Span],
        start: int, end: int,
    ) -> bool:
        if len(spans) <= 1:
            return False
        return max(e - s for s, e in spans) < end - start

    def _recurse(self, spans: List[Span]) -> List[Span]:
        result: List[Span] = []
        for s, e in spans:
            if e - s <= self._max_chunk_size:
                result.append((s, e))
            else:
                result.extend(self._fallback(s, e))
        return result

    def _spans(
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
