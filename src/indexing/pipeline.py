"""Corpus chunking pipeline dispatching files to extension chunkers."""

import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .types import ChunkRecord
from .reader import CorpusReader
from .manifest import ManifestBuilder
from ..chunking.base import BaseChunker
from ..chunking import MarkdownChunker, PythonChunker, TextChunker


class ChunkingPipeline:
    """Coordinate reading, chunking, and auditing across document files."""

    def __init__(self, max_chunk_size: int, raw_dir: str) -> None:
        """Initialize chunking pipeline with chunkers and corpus reader.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
            raw_dir: Corpus root directory containing raw document files.
        """
        self.__max: int = max_chunk_size
        self.__chunkers: Dict[str, BaseChunker] = {
            ".py": PythonChunker(max_chunk_size),
            ".md": MarkdownChunker(max_chunk_size),
            ".txt": TextChunker(max_chunk_size)
        }
        self.__reader: CorpusReader = CorpusReader(
            raw_dir, list(self.__chunkers)
        )

    def discover_files(self) -> List[str]:
        """Return all supported document file paths found in corpus.

        Returns:
            Sorted list of discovered document file paths.
        """
        return self.__reader.discover_files()

    def chunk_corpus(
        self, manifest: ManifestBuilder
    ) -> List[ChunkRecord]:
        """Chunk the entire corpus and record file manifests.

        Args:
            manifest: ManifestBuilder instance to record file hashes.

        Returns:
            List of all chunk records created across the corpus.
        """
        chunks: List[ChunkRecord] = []
        for fp in tqdm(
            self.__reader.discover_files(),
            desc="Chunking", unit="file", leave=False
        ):
            chunks.extend(self.chunk_file(fp, manifest))
        self.audit(chunks)
        return chunks

    def chunk_file(
        self, fp: str, manifest: ManifestBuilder
    ) -> List[ChunkRecord]:
        """Read and chunk a document using its extension-specific chunker.

        Args:
            fp: Path to target document file on disk.
            manifest: ManifestBuilder instance to update.

        Returns:
            List of ChunkRecord dictionaries extracted from the file.
        """
        try:
            content: str = self.__reader.read_file(fp)
        except OSError as e:
            print(f"Warning: {e}", file=sys.stderr)
            return []
        if not content:
            manifest.record(fp, content)
            return []
        chunker: BaseChunker = self.__chunkers[Path(fp).suffix]
        records: List[ChunkRecord] = [
            {
                "source": s,
                "text": content[s.first_character_index:s.last_character_index]
            }
            for s in chunker.chunk(fp, content)
        ]
        manifest.record(fp, content)
        return records

    def audit(self, chunks: List[ChunkRecord]) -> None:
        """Verify chunk size bounds and print sanity check status.

        Args:
            chunks: List of ChunkRecord dictionaries to validate.
        """
        invalid: int = sum(
            not 0
            < c["source"].last_character_index
            - c["source"].first_character_index
            <= self.__max
            for c in chunks
        )
        status: str = "PASS" if not invalid else "FAIL"
        print(f"{len(chunks)} chunks, Sanity: {status}")
