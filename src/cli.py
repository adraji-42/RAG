import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Type

from .models import MinimalSource
from .chunking.base import BaseChunker
from .chunking.markdown import MarkdownChunker
from .chunking.python import PythonChunker


_EXT_MAP: Dict[str, Type[BaseChunker]] = {
    ".py": PythonChunker,
    ".md": MarkdownChunker,
    ".txt": MarkdownChunker
}


class ChunkingPipeline:

    def __init__(
        self,
        max_chunk_size: int,
        raw_dir: str,
    ) -> None:
        self._max_chunk_size: int = max_chunk_size
        self._raw_dir: str = raw_dir
        self._chunkers: Dict[str, BaseChunker] = (
            self._build_chunkers()
        )

    def _build_chunkers(
        self,
    ) -> Dict[str, BaseChunker]:
        return {
            ext: cls(self._max_chunk_size)
            for ext, cls in _EXT_MAP.items()
        }

    def run(self) -> List[MinimalSource]:
        files: List[str] = self._discover_files()
        chunks: List[MinimalSource] = (
            self._process_files(files)
        )
        self._report(len(files), chunks)
        return chunks

    def _discover_files(self) -> List[str]:
        found: List[str] = []
        for root, _, filenames in os.walk(self._raw_dir):
            for name in filenames:
                ext: str = Path(name).suffix
                if ext in self._chunkers:
                    path: str = os.path.join(root, name)
                    found.append(path)
        return found

    def _process_files(
        self,
        files: List[str],
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        bar: tqdm[str] = tqdm(
            files,
            desc="Chunking",
            unit="file",
        )
        for file_path in bar:
            result: List[MinimalSource] = (
                self._process_one(file_path)
            )
            chunks.extend(result)
        return chunks

    def _process_one(
        self,
        file_path: str,
    ) -> List[MinimalSource]:
        content: str = self._read_file(file_path)
        if not content:
            return []
        ext: str = Path(file_path).suffix
        chunker: BaseChunker = self._chunkers[ext]
        return chunker.chunk(file_path, content)

    def _read_file(self, file_path: str) -> str:
        try:
            with open(
                file_path, "r", encoding="utf-8",
                errors="replace",
            ) as fh:
                return fh.read()
        except OSError:
            return ""

    def _report(
        self,
        total_files: int,
        chunks: List[MinimalSource],
    ) -> None:
        invalid: int = self._count_invalid(chunks)
        status: str = self._sanity_status(invalid)
        print(f"Files scanned: {total_files}")
        print(f"Chunks generated: {len(chunks)}")
        print(f"Sanity: {status}")

    def _count_invalid(
        self,
        chunks: List[MinimalSource],
    ) -> int:
        count: int = 0
        for chunk in chunks:
            size: int = (
                chunk.last_character_index
                - chunk.first_character_index
            )
            if size > self._max_chunk_size or size <= 0:
                count += 1
        return count

    def _sanity_status(self, invalid: int) -> str:
        if invalid == 0:
            return "PASS (all chunks valid)"
        return f"FAIL ({invalid} invalid chunks)"


class Cli:

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: Path = Path("data/raw").absolute(),
    ) -> None:
        pipeline: ChunkingPipeline = ChunkingPipeline(
            max_chunk_size=max_chunk_size,
            raw_dir=raw_dir,
        )
        pipeline.run()
