from typing import List, Tuple
from abc import ABC, abstractmethod

from ..models import MinimalSource

Span = Tuple[int, int]


class BaseChunker(ABC):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:

        from .emitter import Emitter

        self.__max_chunk_size: int = max_chunk_size
        self.__emitter: Emitter = Emitter(max_chunk_size)

    @property
    def max_chunk_size(self) -> int:
        return self.__max_chunk_size

    def emit(
        self, fp: str, content: str, spans: List[Span],
    ) -> List[MinimalSource]:
        return self.__emitter.emit(fp, content, spans)

    @abstractmethod
    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        ...
