import os
from pathlib import Path
from typing import Dict, List, Type

from tqdm import tqdm

from ..chunking.base import BaseChunker
from ..chunking.markdown import MarkdownChunker
from ..chunking.python import PythonChunker
from ..models import MinimalSource

_EXT_MAP: Dict[str, Type[BaseChunker]] = {
    ".py": PythonChunker,
    ".md": MarkdownChunker,
}


class Indexer:

    def __init__(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        self._max_chunk_size: int = max_chunk_size
        self._raw_dir: str = raw_dir
        self._processed_dir: str = processed_dir
        self._chunkers: Dict[str, BaseChunker] = {
            ext: cls(max_chunk_size)
            for ext, cls in _EXT_MAP.items()
        }
        self._chunks: List[MinimalSource] = []

    def run(self) -> List[MinimalSource]:
        os.makedirs(self._processed_dir, exist_ok=True)
        files: List[str] = self._discover_files()
        self._chunks = self._chunk_files(files)
        invalid: int = sum(
            1 for c in self._chunks
            if not 0 < (
                c.last_character_index
                - c.first_character_index
            ) <= self._max_chunk_size
        )
        status: str = (
            "PASS (all chunks valid)"
            if invalid == 0
            else f"FAIL ({invalid} invalid chunks)"
        )
        print(f"Files scanned: {len(files)}")
        print(f"Chunks generated: {len(self._chunks)}")
        print(f"Sanity: {status}")
        return self._chunks

    def _discover_files(self) -> List[str]:
        found: List[str] = []
        for root, dirs, names in os.walk(self._raw_dir):
            dirs.sort()
            for name in sorted(names):
                if Path(name).suffix in self._chunkers:
                    found.append(
                        os.path.normpath(
                            os.path.join(root, name),
                        ),
                    )
        return found

    def _chunk_files(
        self, files: List[str],
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        for file_path in tqdm(
            files, desc="Chunking", unit="file",
        ):
            content: str = self._read_file(file_path)
            if not content:
                continue
            chunker: BaseChunker = self._chunkers[
                Path(file_path).suffix
            ]
            chunks.extend(
                chunker.chunk(file_path, content),
            )
        return chunks

    def _read_file(self, file_path: str) -> str:
        try:
            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as handle:
                return handle.read()
        except OSError:
            return ""
