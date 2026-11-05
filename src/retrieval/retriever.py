"""Lexical BM25 retriever querying precomputed disk-backed indices."""

from typing import List, Optional, Tuple

from .cache import QueryCache
from .base import BaseRetriever
from ..models import MinimalSource
from ..indexing.bm25 import BM25Index
from ..indexing.types import ChunkRecord
from ..indexing.storage import IndexStorage


class BM25Retriever(BaseRetriever):
    """Lexical retriever querying disk-serialized BM25 indices."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Initialize the BM25 retriever with storage path and query cache.

        Args:
            processed_dir: Path to directory containing index artifacts.
        """
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__cache: QueryCache = QueryCache(processed_dir)
        self.__chunks: Optional[List[ChunkRecord]] = None
        self.__bm25: Optional[BM25Index] = None

    def __ensure_loaded(self) -> None:
        """Load chunks and BM25 index from storage if not already loaded.

        Raises:
            FileNotFoundError: If index is not found in storage.
        """
        if not self.__storage.has_index():
            raise FileNotFoundError(
                f"Index not found in '{self.__storage.processed_dir}'. "
                "Run 'index' first."
            )
        if self.__chunks is None:
            self.__chunks = self.__storage.load_chunks()
        if self.__bm25 is None:
            self.__bm25 = self.__storage.load_bm25()

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Retrieve top-k sources matching query using BM25 lexical ranking.

        Args:
            query: Search query text string.
            k: Maximum number of sources to return.

        Returns:
            List of retrieved MinimalSource objects.

        Raises:
            ValueError: If k <= 0 or query is empty or whitespace.
        """
        if k <= 0 or not query.strip():
            raise ValueError(
                "k must be greater than 0, "
                "and query must not be empty or contain only whitespace"
            )
        cached: Optional[List[MinimalSource]] = self.__cache.get_sources(
            "bm25", query, k
        )
        if cached is not None:
            return cached
        self.__ensure_loaded()
        if not self.__chunks or self.__bm25 is None:
            return []
        ranked: List[Tuple[int, float]] = self.__bm25.search(query, k=k)
        sources: List[MinimalSource] = [
            self.__chunks[idx]["source"] for idx, _ in ranked
        ]
        self.__cache.set_sources("bm25", query, k, sources)
        return sources
