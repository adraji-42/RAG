"""High-level indexer coordinating corpus chunking and artifact storage."""

import sys
from typing import List, Optional

import numpy as np

from .types import ChunkRecord
from .storage import IndexStorage
from .manifest import ManifestBuilder
from .pipeline import ChunkingPipeline
from .incremental import IncrementalIndexer
from .embeddings import EmbeddingModel, VectorIndex


class Indexer:
    """Coordinate full and incremental indexing of documents into storage."""

    def __init__(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Initialize the indexer with pipeline and storage directories.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
            raw_dir: Directory containing raw corpus document files.
            processed_dir: Output directory for index artifacts.

        Raises:
            ValueError: If max_chunk_size <= 0.
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size > 0 is required")
        self.__pipeline: ChunkingPipeline = ChunkingPipeline(
            max_chunk_size, raw_dir
        )
        self.__storage: IndexStorage = IndexStorage(processed_dir)

    def run(self, force: bool = False, embed: bool = False) -> None:
        """Execute corpus chunking and save artifacts to storage.

        Args:
            force: Force full corpus re-indexing from scratch.
            embed: Whether to compute dense vector embeddings.
        """
        can_incr: bool = not embed or self.__storage.has_embeddings()
        if not force and self.__storage.has_index() and can_incr:
            try:
                return IncrementalIndexer(
                    self.__pipeline, self.__storage
                ).run(embed=embed)
            except ValueError:
                print(
                    "Warning: index out of sync, reindexing from scratch...",
                    file=sys.stderr,
                )
        manifest: ManifestBuilder = ManifestBuilder()
        chunks: List[ChunkRecord] = self.__pipeline.chunk_corpus(manifest)
        vector_index: Optional[VectorIndex] = None
        if embed:
            model: EmbeddingModel = EmbeddingModel()
            embeddings: np.ndarray = model.encode([c["text"] for c in chunks])
            vector_index = VectorIndex(embeddings)
        self.__storage.save_artifacts(chunks, manifest.manifest, vector_index)
        print(
            f"Ingestion complete! Indexed {len(chunks)} "
            f"chunks under {self.__storage.processed_dir}"
        )
