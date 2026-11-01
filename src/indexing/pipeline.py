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

    def __init__(self, max_chunk_size: int, raw_dir: str) -> None:
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
        return self.__reader.discover_files()

    def chunk_corpus(
        self, manifest: ManifestBuilder
    ) -> List[ChunkRecord]:
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
        invalid: int = sum(
            not 0
            < c["source"].last_character_index
            - c["source"].first_character_index
            <= self.__max
            for c in chunks
        )
        status: str = "PASS" if not invalid else "FAIL"
        print(f"{len(chunks)} chunks, Sanity: {status}")
