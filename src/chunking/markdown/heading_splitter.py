from typing import List, Tuple

from ..base import Span
from .block_isolator import BlockIsolator


class HeadingSplitter:

    def __init__(
        self, isolator: BlockIsolator,
        max_chunk_size: int,
    ) -> None:
        self.__isolator: BlockIsolator = isolator
        self.__max: int = max_chunk_size

    def split(
        self, content: str,
        start: int, end: int, plvl: int,
        heads: List[Tuple[int, int]],
        blocks: List[Tuple[str, Span]],
    ) -> List[Span]:
        if end - start <= self.__max:
            return [(start, end)]
        pts: List[int] = self._heading_pts(
            heads, start, end, plvl,
        )
        if pts:
            return self._split_at(
                content, start, end, pts,
                plvl, heads, blocks,
            )
        return self.__isolator.isolate(
            content, start, end, blocks,
        )

    def _heading_pts(
        self, heads: List[Tuple[int, int]],
        start: int, end: int, plvl: int,
    ) -> List[int]:
        cands: List[Tuple[int, int]] = [
            (lv, off) for lv, off in heads
            if start < off < end and lv > plvl
        ]
        if not cands:
            return []
        best: int = min(lv for lv, _ in cands)
        return sorted(
            off for lv, off in cands if lv == best
        )

    def _split_at(
        self, content: str,
        start: int, end: int, pts: List[int],
        plvl: int,
        heads: List[Tuple[int, int]],
        blocks: List[Tuple[str, Span]],
    ) -> List[Span]:
        best: int = min(
            lv for lv, off in heads if off in pts
        )
        spans: List[Span] = []
        prev: int = start
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
