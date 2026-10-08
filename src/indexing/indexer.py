import os
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Type, TypedDict

from ..models import MinimalSource
from .manifest import ManifestBuilder
from .bm25 import BM25Index
from ..chunking.base import BaseChunker
from ..chunking.text import TextChunker
from ..chunking.python import PythonChunker
from ..chunking.markdown import MarkdownChunker


_EXT_MAP: Dict[str, Type[BaseChunker]] = {
    ".py": PythonChunker,
    ".md": MarkdownChunker,
    ".txt": TextChunker,
}


class ChunkRecord(TypedDict):
    source: MinimalSource
    text: str


class Indexer:

    def __init__(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size > 0 is required")
        self.__max_chunk_size: int = max_chunk_size
        self.__raw_dir: str = raw_dir
        self.__processed_dir: str = processed_dir
        self.__chunkers: Dict[str, BaseChunker] = {
            ext: cls(max_chunk_size) for ext, cls in _EXT_MAP.items()
        }
        self.__chunks: List[ChunkRecord] = []
        self.__manifest_builder: ManifestBuilder = ManifestBuilder()

    def run(self) -> List[ChunkRecord]:
        os.makedirs(self.__processed_dir, exist_ok=True)
        files: List[str] = self.__discover_files()
        self.__chunks = self.__process_files(files)
        invalid: int = sum(
            1 for c in self.__chunks
            if not 0 < (
                c["source"].last_character_index
                - c["source"].first_character_index
            ) <= self.__max_chunk_size
        )
        status: str = (
            "PASS (all chunks valid)"
            if invalid == 0
            else f"FAIL ({invalid} invalid chunks)"
        )
        print(f"Chunks generated: {len(self.__chunks)}")
        print(f"Sanity: {status}")
        chunks_fp: str = os.path.join(
            self.__processed_dir, "chunks.pkl",
        )
        manifest_fp: str = os.path.join(
            self.__processed_dir, "manifest.pkl",
        )
        bm25_fp: str = os.path.join(
            self.__processed_dir, "bm25.pkl",
        )
        with open(chunks_fp, "wb") as f:
            pickle.dump(
                self.__chunks, f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        self.__manifest_builder.save(manifest_fp)
        corpus: list[str] = [c["text"] for c in self.__chunks]
        bm25: BM25Index = BM25Index()
        bm25.fit(corpus)
        bm25.save(bm25_fp)
        print(
            f"Saved {len(self.__chunks)} chunks to {chunks_fp} "
            f"and manifest to {manifest_fp}"
        )
        print(
            f"Saved BM25 index with {len(self.__chunks)} chunks "
            f"to {bm25_fp}"
        )
        return self.__chunks

    def __discover_files(self) -> List[str]:
        found: List[str] = []
        for root, _, names in os.walk(self.__raw_dir):
            for name in names:
                if Path(name).suffix in self.__chunkers:
                    found.append(os.path.normpath(os.path.join(root, name)))
        return found

    def __process_files(self, files: List[str]) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for fp in tqdm(files, desc="Chunking", unit="file"):
            content: str = self.__read_file(fp)
            chunker: BaseChunker = self.__chunkers[Path(fp).suffix]
            file_chunks: List[ChunkRecord] = []
            if content:
                for s in chunker.chunk(fp, content):
                    text: str = content[
                        s.first_character_index:s.last_character_index
                    ]
                    file_chunks.append({"source": s, "text": text})
            chunks.extend(file_chunks)
            self.__manifest_builder.record(fp, content, len(file_chunks))
        return chunks

    def __read_file(self, file_path: str) -> str:
        try:
            with open(
                file_path, "r", encoding="utf-8", errors="replace", newline=""
            ) as f:
                return f.read()
        except OSError:
            return ""
