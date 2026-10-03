import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Type

from ..chunking.base import BaseChunker
from ..chunking.text import TextChunker
from ..chunking.python import PythonChunker
from ..chunking.markdown import MarkdownChunker
from ..models import MinimalSource


_EXT_MAP: Dict[str, Type[BaseChunker]] = {
    ".py": PythonChunker,
    ".md": MarkdownChunker,
    ".txt": TextChunker,
}


class Indexer:

    def __init__(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        self.__max_chunk_size: int = max_chunk_size
        self.__raw_dir: str = raw_dir
        self.__processed_dir: str = processed_dir
        self.__chunkers: Dict[str, BaseChunker] = {
            ext: cls(max_chunk_size)
            for ext, cls in _EXT_MAP.items()
        }
        self.__chunks: List[MinimalSource] = []

    @property
    def max_chunk_size(self) -> int:
        return self.__max_chunk_size

    @property
    def raw_dir(self) -> str:
        return self.__raw_dir

    @property
    def processed_dir(self) -> str:
        return self.__processed_dir

    @property
    def chunkers(self) -> Dict[str, BaseChunker]:
        return self.__chunkers

    @property
    def chunks(self) -> List[MinimalSource]:
        return self.__chunks

    def run(self) -> List[MinimalSource]:
        os.makedirs(self.__processed_dir, exist_ok=True)
        files: List[str] = self.__discover_files()
        self.__chunks = self.__chunk_files(files)
        invalid: int = sum(
            1 for c in self.__chunks
            if not 0 < (
                c.last_character_index
                - c.first_character_index
            ) <= self.__max_chunk_size
        )
        status: str = (
            "PASS (all chunks valid)"
            if invalid == 0
            else f"FAIL ({invalid} invalid chunks)"
        )
        print(f"Chunks generated: {len(self.__chunks)}")
        print(f"Sanity: {status}")
        return self.__chunks

    def __discover_files(self) -> List[str]:
        found: List[str] = []
        for root, _, names in os.walk(self.__raw_dir):
            for name in names:
                if Path(name).suffix in self.__chunkers:
                    found.append(os.path.normpath(os.path.join(root, name)))
        return found

    def __chunk_files(
        self, files: List[str],
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        for file_path in tqdm(
            files, desc="Chunking", unit="file",
        ):
            content: str = self.__read_file(file_path)
            if not content:
                continue
            chunker: BaseChunker = self.__chunkers[
                Path(file_path).suffix
            ]
            chunks.extend(
                chunker.chunk(file_path, content),
            )
        return chunks

    def __read_file(self, file_path: str) -> str:
        try:
            with open(
                file_path, "r", encoding="utf-8", errors="replace", newline=""
            ) as file:
                return file.read()
        except OSError:
            return ""
