from pathlib import Path
from typing import Dict, List, Type

from tqdm import tqdm

from .manifest import ManifestBuilder
from .reader import CorpusReader
from .types import ChunkRecord
from ..chunking.base import BaseChunker
from ..chunking import MarkdownChunker, PythonChunker, TextChunker


class ChunkingPipeline:
    _EXT: Dict[str, Type[BaseChunker]] = {
        ".py": PythonChunker,
        ".md": MarkdownChunker,
        ".txt": TextChunker,
    }

    def __init__(self, max_chunk_size: int, raw_dir: str) -> None:
        self.__max: int = max_chunk_size
        self.__reader: CorpusReader = CorpusReader(
            raw_dir, list(self._EXT)
        )
        self.__chunkers: Dict[str, BaseChunker] = {
            k: v(max_chunk_size) for k, v in self._EXT.items()
        }

    def chunk_corpus(
        self, manifest: ManifestBuilder
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for fp in tqdm(
            self.__reader.discover_files(), desc="Chunking", unit="file"
        ):
            chunks.extend(self.__chunk_file(fp, manifest))
        self.__audit(chunks)
        return chunks

    def __chunk_file(
        self, fp: str, manifest: ManifestBuilder
    ) -> List[ChunkRecord]:
        content: str = self.__reader.read_file(fp)
        if not content:
            manifest.record(fp, content, 0)
            return []
        chunker: BaseChunker = self.__chunkers[Path(fp).suffix]
        records: List[ChunkRecord] = [
            {
                "source": s,
                "text": content[
                    s.first_character_index:s.last_character_index
                ],
            }
            for s in chunker.chunk(fp, content)
        ]
        manifest.record(fp, content, len(records))
        return records

    def __audit(self, chunks: List[ChunkRecord]) -> None:
        invalid: int = sum(
            not 0
            < c["source"].last_character_index
            - c["source"].first_character_index
            <= self.__max
            for c in chunks
        )
        status: str = "PASS" if not invalid else "FAIL"
        print(f"{len(chunks)} chunks, Sanity: {status}")
