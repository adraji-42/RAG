from pathlib import Path
from typing import Dict, List, Type

from tqdm import tqdm

from .bm25 import BM25Index
from .types import ChunkRecord
from .reader import CorpusReader
from .storage import IndexStorage
from .manifest import ManifestBuilder
from ..chunking.base import BaseChunker
from ..chunking import MarkdownChunker, PythonChunker, TextChunker


class Indexer:
    _EXT: Dict[str, Type[BaseChunker]] = {
        ".py": PythonChunker, ".md": MarkdownChunker, ".txt": TextChunker
    }

    def __init__(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed"
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size > 0 is required")
        self.__max: int = max_chunk_size
        self.__reader: CorpusReader = CorpusReader(raw_dir, list(self._EXT))
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__manifest: ManifestBuilder = ManifestBuilder()
        self.__chunkers: Dict[str, BaseChunker] = {
            k: v(max_chunk_size) for k, v in self._EXT.items()
        }

    def run(self) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for fp in tqdm(self.__reader.discover_files(), desc="Chunking"):
            c: str = self.__reader.read_file(fp)
            fc: List[ChunkRecord] = [
                {"source": s, "text": c[s.first_character_index:
                                        s.last_character_index]}
                for s in self.__chunkers[Path(fp).suffix].chunk(fp, c)
            ] if c else []
            chunks.extend(fc)
            self.__manifest.record(fp, c, len(fc))
        inv: int = sum(
            not 0 < c["source"].last_character_index
            - c["source"].first_character_index <= self.__max for c in chunks
        )
        print(f"{len(chunks)} chunks, Sanity: {'PASS' if not inv else 'FAIL'}")
        self.__storage.save_chunks(chunks)
        self.__manifest.save(self.__storage.manifest_path)
        (bm25 := BM25Index()).fit([c["text"] for c in chunks])
        bm25.save(self.__storage.bm25_path)
        return chunks
