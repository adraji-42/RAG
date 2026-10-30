from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cache import QueryCache
from .base import BaseRetriever
from ..models import MinimalSource
from ..indexing.bm25 import BM25Index
from ..indexing.types import ChunkRecord
from ..indexing.storage import IndexStorage
from ..indexing.embeddings import EmbeddingModel, VectorIndex


class HybridRetriever(BaseRetriever):
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__cache: QueryCache = QueryCache(processed_dir)
        self.__chunks: Optional[List[ChunkRecord]] = None
        self.__bm25: Optional[BM25Index] = None
        self.__vector: Optional[VectorIndex] = None
        self.__model: Optional[EmbeddingModel] = None

    def __ensure_loaded(self) -> None:
        if self.__chunks is None:
            self.__chunks = self.__storage.load_chunks()
        if self.__bm25 is None:
            self.__bm25 = self.__storage.load_bm25()
        if self.__vector is None:
            self.__vector = self.__storage.load_embeddings()
        if self.__model is None:
            self.__model = EmbeddingModel()

    def __fuse(
        self,
        lexical: List[Tuple[int, float]],
        semantic: List[Tuple[int, float]],
        k: int,
        rrf_k: float = 60.0,
    ) -> List[int]:
        scores: Dict[int, float] = defaultdict(float)
        for rank, (idx, _) in enumerate(lexical, 1):
            scores[idx] += 1.0 / (rrf_k + rank)
        for rank, (idx, _) in enumerate(semantic, 1):
            scores[idx] += 1.0 / (rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [idx for idx, _ in ranked]

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        if k <= 0 or not query.strip():
            return []
        cached: Optional[List[MinimalSource]] = self.__cache.get(
            "hybrid", query, k
        )
        if cached is not None:
            return cached
        self.__ensure_loaded()
        if not (
            self.__chunks and self.__bm25 and self.__vector and self.__model
        ):
            return []
        cand_k: int = max(k * 2, 10)
        lexical = self.__bm25.search(query, k=cand_k)
        query_vec: np.ndarray = self.__model.encode_query(query)
        semantic = self.__vector.search(query_vec, k=cand_k)
        fused: List[int] = self.__fuse(lexical, semantic, k)
        sources: List[MinimalSource] = [
            self.__chunks[i]["source"] for i in fused
        ]
        self.__cache.set("hybrid", query, k, sources)
        return sources
