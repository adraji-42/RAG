from .storage import IndexStorage
from .pipeline import ChunkingPipeline
from .incremental import IncrementalIndexer


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
        self.__incremental.run(force=force)
