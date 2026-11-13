"""CLI command executing full and incremental corpus indexing."""

import sys

from pydantic import ValidationError

from .base import BaseCommand
from ..indexing import Indexer
from .models import IndexOptions


class IndexCommand(BaseCommand):
    """Execute corpus indexing via the command-line interface."""

    def cmd(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        force: bool = False,
        embed: bool = False,
    ) -> None:
        """Run full or incremental indexing with optional embeddings.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
            raw_dir: Corpus directory path containing raw documents.
            processed_dir: Output directory path for processed index files.
            force: Force full corpus re-indexing from scratch.
            embed: Whether to compute dense vector embeddings.
        """
        try:
            options: IndexOptions = IndexOptions(
                max_chunk_size=max_chunk_size,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                force=force,
                embed=embed,
            )
            indexer: Indexer = Indexer(
                options.max_chunk_size,
                options.raw_dir,
                options.processed_dir,
            )
            indexer.run(force=options.force, embed=options.embed)
        except ValidationError as e:
            print(self.format_validation_error(e), file=sys.stderr)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
