from .indexing import Indexer


class Cli:

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        indexer: Indexer = Indexer(
            max_chunk_size=max_chunk_size,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
        indexer.run()
