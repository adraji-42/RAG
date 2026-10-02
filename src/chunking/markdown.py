from typing import List, Tuple
from markdown_it import MarkdownIt
from markdown_it.token import Token

from .base import BaseChunker, Span
from .line_map import LineMap
from ..models import MinimalSource


class MarkdownChunker(BaseChunker):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        super().__init__(max_chunk_size)
        self.__md: MarkdownIt = MarkdownIt()
        self.__content: str
        self.__lmap: LineMap
        self.__headings: List[Tuple[int, int]]
        self.__fences: List[Span]

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        self.__content = content
        self.__lmap = LineMap(content)
        tokens: List[Token] = self.__md.parse(self.__content)
        self.__headings = []
        self.__fences = []
        for tok in tokens:
            if tok.type == "heading_open" and tok.map:
                lv: int = int(tok.tag[1:])
                off: int = self.__lmap.line_start(
                    tok.map[0],
                )
                self.__headings.append((lv, off))
            elif tok.type in ("fence", "code_block"):
                if tok.map:
                    s: int = self.__lmap.line_start(tok.map[0])
                    e: int = self.__lmap.line_start(tok.map[1])
                    self.__fences.append((s, e))
        spans: List[Span] = self.__split(
            0, len(self.__content), 0,
        )
        return self.emit(file_path, content, spans)

    def __split(
        self, start: int, end: int, plvl: int,
    ) -> List[Span]:
        if end - start <= self.max_chunk_size:
            return [(start, end)]
        pts: List[int] = self.__heading_pts(
            start, end, plvl,
        )
        if pts:
            return self.__split_at_pts(
                start, end, pts, plvl,
            )
        return self.__fallback(start, end)

    def __heading_pts(
        self, start: int, end: int, plvl: int,
    ) -> List[int]:
        cands: List[Tuple[int, int]] = [
            (lv, off) for lv, off in self.__headings
            if start < off < end and lv > plvl
        ]
        if not cands:
            return []
        best: int = min(lv for lv, _ in cands)
        return [off for lv, off in cands if lv == best]

    def __split_at_pts(
        self, start: int, end: int,
        pts: List[int], plvl: int,
    ) -> List[Span]:
        best: int = min(
            lv for lv, off in self.__headings
            if off in pts
        )
        spans: List[Span] = []
        prev: int = start
        for p in pts:
            if p > prev:
                spans.extend(
                    self.__split(prev, p, best),
                )
            prev = p
        if prev < end:
            spans.extend(
                self.__split(prev, end, best),
            )
        return spans

    def __fallback(
        self, start: int, end: int,
    ) -> List[Span]:
        fb: List[Span] = [
            (s, e) for s, e in self.__fences
            if start < s < end or start < e < end
        ]
        if fb:
            fs = self.__fence_spans(fb, start, end)
            if self.__is_useful(fs, start, end):
                return self.__recurse_fallback(fs)
        ps: List[Span] = self.split_paragraphs(
            self.__content, start, end,
        )
        if self.__is_useful(ps, start, end):
            return self.__recurse_fallback(ps)
        return [(start, end)]

    def __fence_spans(
        self, fb: List[Span],
        start: int, end: int,
    ) -> List[Span]:
        bounds: List[int] = []
        for s, e in fb:
            if s > start:
                bounds.append(s)
            if e < end:
                bounds.append(e)
        spans: List[Span] = []
        prev: int = start
        for b in sorted(set(bounds)):
            if b > prev:
                spans.append((prev, b))
            prev = b
        if prev < end:
            spans.append((prev, end))
        return spans

    def __is_useful(
        self, spans: List[Span],
        start: int, end: int,
    ) -> bool:
        if len(spans) <= 1:
            return False
        return (
            max(e - s for s, e in spans) < end - start
        )

    def __recurse_fallback(
        self, spans: List[Span],
    ) -> List[Span]:
        result: List[Span] = []
        for s, e in spans:
            if e - s <= self.max_chunk_size:
                result.append((s, e))
            else:
                result.extend(self.__fallback(s, e))
        return result
