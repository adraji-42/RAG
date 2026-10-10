from typing import List

from .bm25 import BM25Index
from .manifest import ManifestBuilder
from .pipeline import ChunkingPipeline
from .storage import IndexStorage
from .types import ChunkRecord


class Indexer:
    _EXT = ChunkingPipeline._EXT

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
        self.__manifest: ManifestBuilder = ManifestBuilder()

    def run(self) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = self.__pipeline.chunk_corpus(
            self.__manifest
        )
        self.__storage.save_chunks(chunks)
        self.__storage.save_manifest(self.__manifest.manifest)
        bm25: BM25Index = BM25Index([c["text"] for c in chunks])
        self.__storage.save_bm25(bm25)
        return chunks
