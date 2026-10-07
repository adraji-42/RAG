from typing import List

from ..base import BaseChunker, Span
from ...models import MinimalSource
from .splitter import RecursiveTextSplitter


class TextChunker(BaseChunker):

    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)
        self.__splitter: RecursiveTextSplitter = RecursiveTextSplitter(
            max_chunk_size,
        )

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        spans: List[Span] = self.split_block(content, 0, len(content))
        return self.emit(file_path, content, spans)

    def split_block(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        return self.__splitter.split_block(content, start, end)

    def split_lines(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        return self.__splitter.split_lines(content, start, end)
