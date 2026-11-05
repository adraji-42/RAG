"""Disk storage manager for serialized index artifacts and cache."""

import os
import pickle
from typing import Any, Dict, List, Optional, cast

import numpy as np

from .bm25 import BM25Index
from .embeddings import VectorIndex
from .types import ChunkRecord, ManifestRecord


class IndexStorage:
    """Manage persistence of serialized index artifacts and cache on disk."""

    def __init__(self, processed_dir: str) -> None:
        """Initialize index storage directory and create it if missing.

        Args:
            processed_dir: Path to directory for serialized index files.
        """
        self.__processed_dir: str = processed_dir
        os.makedirs(self.__processed_dir, exist_ok=True)

    @property
    def processed_dir(self) -> str:
        """Return the root directory storing processed index files.

        Returns:
            Directory path string for index artifacts.
        """
        return self.__processed_dir

    @property
    def chunks_path(self) -> str:
        """Return the filesystem path to chunks.pkl.

        Returns:
            Path string to chunks.pkl.
        """
        return os.path.join(self.__processed_dir, "chunks.pkl")

    @property
    def manifest_path(self) -> str:
        """Return the filesystem path to manifest.pkl.

        Returns:
            Path string to manifest.pkl.
        """
        return os.path.join(self.__processed_dir, "manifest.pkl")

    @property
    def bm25_path(self) -> str:
        """Return the filesystem path to bm25.pkl.

        Returns:
            Path string to bm25.pkl.
        """
        return os.path.join(self.__processed_dir, "bm25.pkl")

    @property
    def embeddings_path(self) -> str:
        """Return the filesystem path to embeddings.pkl.

        Returns:
            Path string to embeddings.pkl.
        """
        return os.path.join(self.__processed_dir, "embeddings.pkl")

    @property
    def cache_path(self) -> str:
        """Return the filesystem path to cache.pkl.

        Returns:
            Path string to cache.pkl.
        """
        return os.path.join(self.__processed_dir, "cache.pkl")

    def __dump(self, path: str, data: Any) -> None:
        """Serialize data to disk using highest pickle protocol.

        Args:
            path: Output file path string.
            data: Object to serialize via pickle.

        Raises:
            OSError: If serializing or writing to disk fails.
        """
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            raise OSError(f"Failed to write file: {path}")

    def __load(self, path: str, default: Any) -> Any:
        """Deserialize data from disk or return default if missing.

        Args:
            path: Input file path string to deserialize.
            default: Fallback default value if file does not exist.

        Returns:
            Deserialized object or default value.

        Raises:
            RuntimeError: If index file is corrupted or unreadable.
        """
        if not os.path.isfile(path):
            return default
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            raise RuntimeError(f"Corrupted or unreadable index file: {path}")

    def has_index(self) -> bool:
        """Return True if chunks, manifest, and BM25 index files exist.

        Returns:
            True if all lexical index artifacts exist, False otherwise.
        """
        return (
            os.path.isfile(self.chunks_path)
            and os.path.isfile(self.manifest_path)
            and os.path.isfile(self.bm25_path)
        )

    def has_embeddings(self) -> bool:
        """Return True if the embeddings file exists on disk.

        Returns:
            True if embeddings.pkl exists, False otherwise.
        """
        return os.path.isfile(self.embeddings_path)

    def save_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Save chunk records to disk.

        Args:
            chunks: List of ChunkRecord dictionaries to serialize.
        """
        self.__dump(self.chunks_path, chunks)

    def load_chunks(self) -> List[ChunkRecord]:
        """Load chunk records from disk.

        Returns:
            List of deserialized ChunkRecord dictionaries.
        """
        return cast(List[ChunkRecord], self.__load(self.chunks_path, []))

    def save_manifest(
        self, manifest: Dict[str, ManifestRecord]
    ) -> None:
        """Save corpus manifest to disk.

        Args:
            manifest: Dictionary mapping file paths to records.
        """
        self.__dump(self.manifest_path, manifest)

    def load_manifest(self) -> Dict[str, ManifestRecord]:
        """Load corpus manifest from disk.

        Returns:
            Dictionary mapping file paths to manifest records.
        """
        return cast(
            Dict[str, ManifestRecord], self.__load(self.manifest_path, {})
        )

    def save_bm25(self, index: BM25Index) -> None:
        """Save BM25 index instance to disk.

        Args:
            index: BM25Index instance to serialize.
        """
        self.__dump(self.bm25_path, index)

    def load_bm25(self) -> BM25Index:
        """Load BM25 index instance from disk.

        Returns:
            Deserialized BM25Index instance.
        """
        return cast(
            BM25Index, self.__load(self.bm25_path, BM25Index([]))
        )

    def save_embeddings(self, index: VectorIndex) -> None:
        """Save vector index instance to disk.

        Args:
            index: VectorIndex instance to serialize.
        """
        self.__dump(self.embeddings_path, index)

    def load_embeddings(self) -> VectorIndex:
        """Load vector index instance from disk.

        Returns:
            Deserialized VectorIndex instance.
        """
        empty: np.ndarray = np.empty((0, 384), dtype=np.float32)
        return cast(
            VectorIndex,
            self.__load(self.embeddings_path, VectorIndex(empty)),
        )

    def cache_mtime(self) -> Optional[float]:
        """Return cache file modification timestamp if present.

        Returns:
            Modification timestamp float of cache.pkl or None if missing.
        """
        if not os.path.isfile(self.cache_path):
            return None
        return os.path.getmtime(self.cache_path)

    def load_cache(self) -> Dict[Any, Any]:
        """Load cached queries and answers from disk.

        Returns:
            Dictionary of cached queries and answers.
        """
        return dict(self.__load(self.cache_path, {}))

    def save_cache(self, cache: Dict[Any, Any]) -> None:
        """Save query and answer cache to disk.

        Args:
            cache: Dictionary of queries and answers to save.
        """
        self.__dump(self.cache_path, cache)

    def save_artifacts(
        self,
        chunks: List[ChunkRecord],
        manifest: Dict[str, ManifestRecord],
        vector_index: Optional[VectorIndex] = None,
    ) -> None:
        """Persist chunks, manifest, BM25, and embeddings to disk.

        Args:
            chunks: List of ChunkRecord dictionaries to serialize.
            manifest: Manifest dictionary to serialize.
            vector_index: Optional VectorIndex instance to serialize.
        """
        self.save_chunks(chunks)
        self.save_manifest(manifest)
        self.save_bm25(BM25Index([c["text"] for c in chunks]))
        if vector_index is not None:
            self.save_embeddings(vector_index)
        if os.path.isfile(self.cache_path):
            os.remove(self.cache_path)
