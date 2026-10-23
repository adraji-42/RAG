from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseRetriever
from ..models import MinimalSource
from ..indexing.bm25 import BM25Index
from ..indexing.types import ChunkRecord
from ..indexing.storage import IndexStorage
from ..indexing.embeddings import EmbeddingModel, VectorIndex


class HybridRetriever(BaseRetriever):
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        try:
            self.__chunks: List[ChunkRecord] = self.__storage.load_chunks()
        except (OSError, EOFError):
            self.__chunks = []
        try:
            self.__bm25: BM25Index = self.__storage.load_bm25()
        except (OSError, EOFError):
            self.__bm25 = BM25Index([])
        try:
            self.__vector: VectorIndex = self.__storage.load_embeddings()
        except (OSError, EOFError):
            self.__vector = VectorIndex(np.empty((0, 384), dtype=np.float32))
        self.__model: EmbeddingModel = EmbeddingModel()

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
        if k <= 0 or not query.strip() or not self.__chunks:
            return []
        candidates_k: int = max(k * 2, 10)
        lexical = self.__bm25.search(query, k=candidates_k)
        query_vec: np.ndarray = self.__model.encode_query(query)
        semantic = self.__vector.search(query_vec, k=candidates_k)
        fused_indices: List[int] = self.__fuse(lexical, semantic, k)
        return [
            self.__chunks[idx]["source"]
            for idx in fused_indices
            if 0 <= idx < len(self.__chunks)
        ]
