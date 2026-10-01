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
        self._chunkers: Dict[str, BaseChunker] = (
            self._build_chunkers()
        )
        self._chunks: List[MinimalSource] = []

    def _build_chunkers(self) -> Dict[str, BaseChunker]:
        return {
            ext: cls(self._max_chunk_size)
            for ext, cls in _EXT_MAP.items()
        }

    def run(self) -> List[MinimalSource]:
        os.makedirs(self._processed_dir, exist_ok=True)
        files: List[str] = self._discover_files()
        self._chunks = self._chunk_files(files)
        self._report(len(files))
        return self._chunks

    def _discover_files(self) -> List[str]:
        found: List[str] = []
        for root, dirs, names in os.walk(self._raw_dir):
            dirs.sort()
            for name in sorted(names):
                if Path(name).suffix in self._chunkers:
                    found.append(self._relative(root, name))
        return found

    def _relative(self, root: str, name: str) -> str:
        return os.path.relpath(
            os.path.join(root, name),
            self._raw_dir,
        )

    def _chunk_files(
        self,
        files: List[str],
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        for rel_path in tqdm(
            files,
            desc="Chunking",
            unit="file",
        ):
            chunks.extend(self._chunk_one(rel_path))
        return chunks

    def _chunk_one(self, rel_path: str) -> List[MinimalSource]:
        content: str = self._read_file(rel_path)
        if not content:
            return []
        chunker: BaseChunker = self._chunkers[
            Path(rel_path).suffix
        ]
        return chunker.chunk(rel_path, content)

    def _read_file(self, rel_path: str) -> str:
        full_path: str = os.path.join(self._raw_dir, rel_path)
        try:
            with open(
                full_path,
                "r",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as handle:
                return handle.read()
        except OSError:
            return ""

    def _is_valid(self, chunk: MinimalSource) -> bool:
        size: int = (
            chunk.last_character_index
            - chunk.first_character_index
        )
        return 0 < size <= self._max_chunk_size

    def _count_invalid(self) -> int:
        return sum(
            1 for chunk in self._chunks
            if not self._is_valid(chunk)
        )

    def _report(self, total_files: int) -> None:
        invalid: int = self._count_invalid()
        status: str = (
            "PASS (all chunks valid)"
            if invalid == 0
            else f"FAIL ({invalid} invalid chunks)"
        )
        print(f"Files scanned: {total_files}")
        print(f"Chunks generated: {len(self._chunks)}")
        print(f"Sanity: {status}")
