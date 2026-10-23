from typing import List

import numpy as np

from .types import ChunkRecord
from .storage import IndexStorage
from .manifest import ManifestBuilder
from .pipeline import ChunkingPipeline
from .incremental import IncrementalIndexer
from .embeddings import EmbeddingModel, VectorIndex


class Indexer:
    def __init__(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size > 0 is required")
        self.__pipeline: ChunkingPipeline = ChunkingPipeline(
            max_chunk_size, raw_dir
        )
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__incremental: IncrementalIndexer = IncrementalIndexer(
            self.__pipeline, self.__storage
        )

    def run(self, force: bool = False) -> None:
        if not force and self.__storage.has_index():
            self.__incremental.run()
            return
        manifest: ManifestBuilder = ManifestBuilder()
        chunks: List[ChunkRecord] = self.__pipeline.chunk_corpus(manifest)
        model: EmbeddingModel = EmbeddingModel()
        embeddings: np.ndarray = model.encode([c["text"] for c in chunks])
        vector_index: VectorIndex = VectorIndex(embeddings)
        self.__storage.save_artifacts(chunks, manifest.manifest, vector_index)
        print(
            f"Ingestion complete! Indexed {len(chunks)} "
            f"chunks under {self.__storage.processed_dir}"
        )
