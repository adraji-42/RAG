import os
from collections import OrderedDict
from typing import List, Optional, Tuple

from ..models import MinimalSource
from ..indexing.storage import IndexStorage


class QueryCache:
    def __init__(
        self, processed_dir: str = "data/processed", max_size: int = 512
    ) -> None:
        self.__storage: IndexStorage = IndexStorage(processed_dir)
        self.__max_size: int = max_size
        self.__cache: OrderedDict[
            Tuple[str, str, int], List[MinimalSource]
        ] = self.__storage.load_cache()

    def get(
        self, mode: str, query: str, k: int
    ) -> Optional[List[MinimalSource]]:
        key: Tuple[str, str, int] = (mode, query.strip(), k)
        if key in self.__cache:
            self.__cache.move_to_end(key)
            return list(self.__cache[key])
        return None

    def set(
        self, mode: str, query: str, k: int, results: List[MinimalSource]
    ) -> None:
        key: Tuple[str, str, int] = (mode, query.strip(), k)
        self.__cache[key] = list(results)
        self.__cache.move_to_end(key)
        if len(self.__cache) > self.__max_size:
            self.__cache.popitem(last=False)
        self.__storage.save_cache(self.__cache)

    def clear(self) -> None:
        self.__cache.clear()
        try:
            if os.path.isfile(self.__storage.cache_path):
                os.remove(self.__storage.cache_path)
        except OSError:
            pass


__all__: List[str] = ["QueryCache"]
