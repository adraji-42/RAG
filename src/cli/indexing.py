from .base import BaseCommand
from ..indexing import Indexer


class IndexCommand(BaseCommand):
    def cmd(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        force: bool = False,
    ) -> None:
        indexer: Indexer = Indexer(
            max_chunk_size=max_chunk_size,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
        indexer.run(force=force)
