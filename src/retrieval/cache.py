"""Disk-backed query and generated answer cache with FIFO eviction."""

import os
from typing import Any, Dict, List, Optional, Tuple

from ..models import MinimalSource
from ..indexing.storage import IndexStorage


class QueryCache:
    """Manage disk-synced FIFO query and answer cache."""

    def __init__(
        self, processed_dir: str = "data/processed", max_size: int = 512
    ) -> None:
        """Initialize the query cache with storage directory and capacity.

        Args:
            processed_dir: Path to storage directory containing cache.
            max_size: Maximum number of cached query entries allowed.
        """
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__max_size: int = max_size
        self.__mtime: Optional[float] = self.__storage.cache_mtime()
        self.__cache: Dict[Any, Any] = self.__storage.load_cache()

    def __reload_if_stale(self) -> None:
        """Reload cache contents from disk if cache file mtime changed."""
        mtime: Optional[float] = self.__storage.cache_mtime()
        if mtime is None or mtime == self.__mtime:
            return
        self.__mtime = mtime
        self.__cache = self.__storage.load_cache()

    def get_sources(
        self, mode: str, query: str, k: int
    ) -> Optional[List[MinimalSource]]:
        """Retrieve cached retrieval sources for a given query and mode.

        Args:
            mode: Retrieval mode ('bm25' or 'hybrid').
            query: Search query text string.
            k: Maximum number of retrieved sources requested.

        Returns:
            Cached list of MinimalSource spans, or None if not cached.
        """
        self.__reload_if_stale()
        key: Tuple[str, str, int] = (mode, query.strip(), k)
        return self.__cache.get(key)

    def set_sources(
        self, mode: str, query: str, k: int, results: List[MinimalSource]
    ) -> None:
        """Store retrieval results in cache and enforce FIFO eviction.

        Args:
            mode: Retrieval mode ('bm25' or 'hybrid').
            query: Search query text string.
            k: Maximum number of retrieved sources requested.
            results: List of MinimalSource objects to cache.
        """
        key: Tuple[str, str, int] = (mode, query.strip(), k)
        self.__cache[key] = list(results)
        if len(self.__cache) > self.__max_size:
            first = next(iter(self.__cache))
            del self.__cache[first]
        self.__storage.save_cache(self.__cache)
        self.__mtime = self.__storage.cache_mtime()

    def get_answer(self, key: str) -> Optional[str]:
        """Retrieve cached generated answer for a given cache key.

        Args:
            key: Unique answer cache key string.

        Returns:
            Cached generated answer string, or None if not cached.
        """
        self.__reload_if_stale()
        return self.__cache.get(("answer", key))

    def set_answer(self, key: str, answer: str) -> None:
        """Store generated answer in cache and enforce FIFO eviction.

        Args:
            key: Unique answer cache key string.
            answer: Generated answer text string to cache.
        """
        self.__cache[("answer", key)] = answer
        if len(self.__cache) > self.__max_size:
            first = next(iter(self.__cache))
            del self.__cache[first]
        self.__storage.save_cache(self.__cache)
        self.__mtime = self.__storage.cache_mtime()

    def clear(self) -> None:
        """Clear in-memory cache and remove the on-disk cache file."""
        self.__cache.clear()
        try:
            if os.path.isfile(self.__storage.cache_path):
                os.remove(self.__storage.cache_path)
        except OSError:
            pass
