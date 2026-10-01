from abc import ABC, abstractmethod
from typing import List, Tuple

from ..models import MinimalSource

Span = Tuple[int, int]


class SpanPacker:

    def __init__(self, max_chunk_size: int) -> None:
        self._max: int = max_chunk_size

    def split_paragraphs(
        self, text: str, start: int, end: int,
    ) -> List[Span]:
        parts: List[str] = text[start:end].split(
            "\n\n",
        )
        if len(parts) <= 1:
            return [(start, end)]
        return self._parts_spans(parts, start, end)

    def _parts_spans(
        self, parts: List[str],
        start: int, end: int,
    ) -> List[Span]:
        spans: List[Span] = []
        pos: int = start
        for part in parts:
            se: int = min(pos + len(part), end)
            if se < end:
                se += 2
            spans.append((pos, min(se, end)))
            pos = min(se, end)
        return spans

    def pack(
        self, content: str, spans: List[Span],
    ) -> List[Span]:
        if not spans:
            return []
        packed: List[Span] = []
        cs, ce = spans[0]
        for s, e in spans[1:]:
            cs, ce = self._try_merge(
                content, packed, cs, ce, s, e,
            )
        packed.append((cs, ce))
        return packed

    def _try_merge(
        self, content: str, packed: List[Span],
        cs: int, ce: int, s: int, e: int,
    ) -> Span:
        ws: bool = content[ce:s].strip() == ""
        if ws and e - cs <= self._max:
            return cs, e
        packed.append((cs, ce))
        return s, e

    def filter_empty(
        self, content: str, spans: List[Span],
    ) -> List[Span]:
        return [
            (s, e) for s, e in spans
            if s < e and content[s:e].strip()
        ]

    def emit(
        self, fp: str, content: str,
        spans: List[Span],
    ) -> List[MinimalSource]:
        clean: List[Span] = self.filter_empty(
            content, spans,
        )
        packed: List[Span] = self.pack(
            content, clean,
        )
        return self._emit_all(fp, content, packed)

    def _emit_all(
        self, fp: str, content: str,
        packed: List[Span],
    ) -> List[MinimalSource]:
        result: List[MinimalSource] = []
        for s, e in packed:
            if e - s <= self._max:
                result.append(self._src(fp, s, e))
            else:
                result.extend(
                    self._split_nl(fp, content, s, e),
                )
        return result

    def _split_nl(
        self, fp: str, content: str,
        start: int, end: int,
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        pos: int = start
        while pos < end:
            cut: int = self._cut(content, pos, end)
            chunks.append(self._src(fp, pos, cut))
            pos = cut
        return chunks

    def _cut(
        self, content: str, pos: int, end: int,
    ) -> int:
        if end - pos <= self._max:
            return end
        limit: int = pos + self._max
        nl: int = content.rfind("\n", pos, limit)
        return nl + 1 if nl > pos else min(limit, end)

    def _src(
        self, fp: str, start: int, end: int,
    ) -> MinimalSource:
        return MinimalSource(
            file_path=fp,
            first_character_index=start,
            last_character_index=end,
        )


class BaseChunker(ABC):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        self._max_chunk_size: int = max_chunk_size
        self._packer: SpanPacker = SpanPacker(
            max_chunk_size,
        )

    @abstractmethod
    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        ...
