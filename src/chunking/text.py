from typing import List

from .base import BaseChunker, Span
from ..models import MinimalSource


class TextChunker(BaseChunker):

    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        spans: List[Span] = self.split_block(content, 0, len(content))
        return self.emit(file_path, content, spans)

    def split_block(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        if end - start <= self.max_chunk_size:
            return [(start, end)]
        spans: List[Span] = self.split_paragraphs(
            content, start, end,
        )
        if self.__all_fit(spans):
            return spans
        refined: List[Span] = []
        for s, e in spans:
            if e - s <= self.max_chunk_size:
                refined.append((s, e))
            else:
                refined.extend(
                    self.split_lines(content, s, e),
                )
        return refined

    def split_lines(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        lines: List[str] = content[start:end].split("\n")
        spans: List[Span] = []
        pos: int = start
        for line in lines:
            le: int = len(line)
            line_end: int = min(pos + le + 1, end)
            spans.append((pos, line_end))
            pos = line_end
        return spans

    def __all_fit(self, spans: List[Span]) -> bool:
        return all(
            e - s <= self.max_chunk_size
            for s, e in spans
        )
