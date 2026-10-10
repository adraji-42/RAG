import os
import pickle
from typing import Dict, List, cast

from .bm25 import BM25Index
from .types import ChunkRecord, ManifestRecord


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

    def save_manifest(
        self, manifest: Dict[str, ManifestRecord]
    ) -> None:
        with open(self.manifest_path, "wb") as f:
            pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_manifest(self) -> Dict[str, ManifestRecord]:
        with open(self.manifest_path, "rb") as f:
            return cast(
                Dict[str, ManifestRecord], pickle.load(f)
            )

    def save_bm25(self, index: BM25Index) -> None:
        with open(self.bm25_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_bm25(self) -> BM25Index:
        with open(self.bm25_path, "rb") as f:
            return cast(BM25Index, pickle.load(f))
