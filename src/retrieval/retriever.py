from typing import List, Tuple

from .base import BaseRetriever
from ..indexing.bm25 import BM25Index
from ..indexing.storage import IndexStorage
from ..indexing.types import ChunkRecord
from ..models import MinimalSource


class BM25Retriever(BaseRetriever):

    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        try:
            self.__chunks: List[ChunkRecord] = (
                self.__storage.load_chunks()
            )
        except (FileNotFoundError, OSError, EOFError):
            self.__chunks = []
        try:
            self.__bm25: BM25Index = self.__storage.load_bm25()
        except (FileNotFoundError, OSError, EOFError):
            self.__bm25 = BM25Index([])

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        if k <= 0 or not query.strip() or not self.__chunks:
            return []
        ranked: List[Tuple[int, float]] = self.__bm25.search(query, k=k)
        return [
            self.__chunks[idx]["source"]
            for idx, _ in ranked
            if 0 <= idx < len(self.__chunks)
        ]
