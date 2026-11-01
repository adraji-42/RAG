import os
import pickle
from typing import Any, Dict, List, Optional, cast

import numpy as np

from .bm25 import BM25Index
from .embeddings import VectorIndex
from .types import ChunkRecord, ManifestRecord


class IndexStorage:
    def __init__(self, processed_dir: str) -> None:
        self.__processed_dir: str = processed_dir
        os.makedirs(self.__processed_dir, exist_ok=True)

    @property
    def processed_dir(self) -> str:
        return self.__processed_dir

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.__processed_dir, "chunks.pkl")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.__processed_dir, "manifest.pkl")

    @property
    def bm25_path(self) -> str:
        return os.path.join(self.__processed_dir, "bm25.pkl")

    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.__processed_dir, "embeddings.pkl")

    @property
    def cache_path(self) -> str:
        return os.path.join(self.__processed_dir, "cache.pkl")

    def __dump(self, path: str, data: Any) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            raise OSError(f"Failed to write file: {path}")

    def __load(self, path: str, default: Any) -> Any:
        if not os.path.isfile(path):
            return default
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            raise RuntimeError(f"Corrupted or unreadable index file: {path}")

    def has_index(self) -> bool:
        return (
            os.path.isfile(self.chunks_path)
            and os.path.isfile(self.manifest_path)
            and os.path.isfile(self.bm25_path)
            and os.path.isfile(self.embeddings_path)
        )

    def save_chunks(self, chunks: List[ChunkRecord]) -> None:
        self.__dump(self.chunks_path, chunks)

    def load_chunks(self) -> List[ChunkRecord]:
        return cast(List[ChunkRecord], self.__load(self.chunks_path, []))

    def save_manifest(
        self, manifest: Dict[str, ManifestRecord]
    ) -> None:
        self.__dump(self.manifest_path, manifest)

    def load_manifest(self) -> Dict[str, ManifestRecord]:
        return cast(
            Dict[str, ManifestRecord], self.__load(self.manifest_path, {})
        )

    def save_bm25(self, index: BM25Index) -> None:
        self.__dump(self.bm25_path, index)

    def load_bm25(self) -> BM25Index:
        return cast(
            BM25Index, self.__load(self.bm25_path, BM25Index([]))
        )

    def save_embeddings(self, index: VectorIndex) -> None:
        self.__dump(self.embeddings_path, index)

    def load_embeddings(self) -> VectorIndex:
        empty: np.ndarray = np.empty((0, 384), dtype=np.float32)
        return cast(
            VectorIndex,
            self.__load(self.embeddings_path, VectorIndex(empty)),
        )

    def cache_mtime(self) -> Optional[float]:
        if not os.path.isfile(self.cache_path):
            return None
        return os.path.getmtime(self.cache_path)

    def load_cache(self) -> Dict[Any, Any]:
        return dict(self.__load(self.cache_path, {}))

    def save_cache(self, cache: Dict[Any, Any]) -> None:
        self.__dump(self.cache_path, cache)

    def save_artifacts(
        self,
        chunks: List[ChunkRecord],
        manifest: Dict[str, ManifestRecord],
        vector_index: VectorIndex,
    ) -> None:
        self.save_chunks(chunks)
        self.save_manifest(manifest)
        self.save_bm25(BM25Index([c["text"] for c in chunks]))
        self.save_embeddings(vector_index)
        if os.path.isfile(self.cache_path):
            os.remove(self.cache_path)
