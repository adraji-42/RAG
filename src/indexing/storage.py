import os
import pickle
from typing import List, cast

from .types import ChunkRecord


class IndexStorage:

    def __init__(self, processed_dir: str) -> None:
        self.__processed_dir: str = processed_dir
        os.makedirs(self.__processed_dir, exist_ok=True)

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.__processed_dir, "chunks.pkl")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.__processed_dir, "manifest.pkl")

    @property
    def bm25_path(self) -> str:
        return os.path.join(self.__processed_dir, "bm25.pkl")

    def save_chunks(self, chunks: List[ChunkRecord]) -> None:
        with open(self.chunks_path, "wb") as f:
            pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_chunks(self) -> List[ChunkRecord]:
        with open(self.chunks_path, "rb") as f:
            return cast(List[ChunkRecord], pickle.load(f))
