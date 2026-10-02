from abc import ABC, abstractmethod
from typing import List, Tuple

from ..models import MinimalSource

Span = Tuple[int, int]


class BaseChunker(ABC):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        self.__max_chunk_size: int = max_chunk_size

    @property
    def max_chunk_size(self) -> int:
        return self.__max_chunk_size

    @abstractmethod
    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        ...

    def split_paragraphs(
        self, text: str, start: int, end: int,
    ) -> List[Span]:
        parts: List[str] = text[start:end].split("\n\n")
        if len(parts) <= 1:
            return [(start, end)]
        spans: List[Span] = []
        pos: int = start
        for part in parts:
            se: int = min(pos + len(part), end)
            if se < end:
                se += 2
            spans.append((pos, min(se, end)))
            pos = min(se, end)
        return spans

    def emit(
        self, fp: str, content: str, spans: List[Span],
    ) -> List[MinimalSource]:
        packed: List[Span] = self.__pack(content, spans)
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
                    self.__split_newlines(fp, content, s, e),
                )
        return result

    def __pack(
        self, content: str, spans: List[Span],
    ) -> List[Span]:
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

    def __split_newlines(
        self, fp: str, content: str,
        start: int, end: int,
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        pos: int = start
        while pos < end:
            if end - pos <= self.__max_chunk_size:
                cut: int = end
            else:
                limit: int = pos + self.__max_chunk_size
                nl: int = content.rfind("\n", pos, limit)
                cut = nl + 1 if nl > pos else min(limit, end)
            chunks.append(MinimalSource(
                file_path=fp,
                first_character_index=pos,
                last_character_index=cut,
            ))
            pos = cut
        return chunks
