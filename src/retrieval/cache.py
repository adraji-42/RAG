import os
from typing import Any, Dict, List, Optional, Tuple, cast

from ..models import MinimalSource
from ..indexing.storage import IndexStorage


class QueryCache:
    def __init__(
        self, processed_dir: str = "data/processed", max_size: int = 512
    ) -> None:
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__max_size: int = max_size
        self.__mtime: Optional[float] = self.__storage.cache_mtime()
        self.__cache: Dict[Any, Any] = dict(
            self.__storage.load_cache()
        )

    def __reload_if_stale(self) -> None:
        mtime: Optional[float] = self.__storage.cache_mtime()
        if mtime is None or mtime == self.__mtime:
            return
        self.__mtime = mtime
        self.__cache = dict(self.__storage.load_cache())

    def get_sources(
        self, mode: str, query: str, k: int
    ) -> Optional[List[MinimalSource]]:
        self.__reload_if_stale()
        key: Tuple[str, str, int] = (mode, query.strip(), k)
        val = self.__cache.get(key)
        if val is None:
            return None
        return cast(List[MinimalSource], list(val))

    def set_sources(
        self, mode: str, query: str, k: int, results: List[MinimalSource]
    ) -> None:
        key: Tuple[str, str, int] = (mode, query.strip(), k)
        self.__cache[key] = list(results)
        if len(self.__cache) > self.__max_size:
            first = next(iter(self.__cache))
            del self.__cache[first]
        self.__storage.save_cache(self.__cache)
        self.__mtime = self.__storage.cache_mtime()

    def get_answer(self, key: str) -> Optional[str]:
        self.__reload_if_stale()
        val = self.__cache.get(("answer", key))
        if val is None:
            return None
        return cast(str, val)

    def set_answer(self, key: str, answer: str) -> None:
        self.__cache[("answer", key)] = answer
        if len(self.__cache) > self.__max_size:
            first = next(iter(self.__cache))
            del self.__cache[first]
        self.__storage.save_cache(self.__cache)
        self.__mtime = self.__storage.cache_mtime()

    def clear(self) -> None:
        self.__cache.clear()
        try:
            if os.path.isfile(self.__storage.cache_path):
                os.remove(self.__storage.cache_path)
        except OSError:
            pass


__all__: List[str] = ["QueryCache"]
