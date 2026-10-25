from typing import List, Optional, Tuple

from .cache import QueryCache
from .base import BaseRetriever
from ..models import MinimalSource
from ..indexing.bm25 import BM25Index
from ..indexing.types import ChunkRecord
from ..indexing.storage import IndexStorage


class BM25Retriever(BaseRetriever):
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__cache: QueryCache = QueryCache(processed_dir)
        self.__chunks: Optional[List[ChunkRecord]] = None
        self.__bm25: Optional[BM25Index] = None

    def __ensure_loaded(self) -> None:
        if self.__chunks is None:
            self.__chunks = self.__storage.load_chunks()
        if self.__bm25 is None:
            self.__bm25 = self.__storage.load_bm25()

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        if k <= 0 or not query.strip():
            return []
        cached: Optional[List[MinimalSource]] = self.__cache.get(
            "bm25", query, k
        )
        if cached is not None:
            return cached
        self.__ensure_loaded()
        if not self.__chunks or self.__bm25 is None:
            return []
        ranked: List[Tuple[int, float]] = self.__bm25.search(query, k=k)
        sources: List[MinimalSource] = [
            self.__chunks[idx]["source"]
            for idx, _ in ranked
            if 0 <= idx < len(self.__chunks)
        ]
        self.__cache.set("bm25", query, k, sources)
        return sources
