from typing import Any
from abc import ABC, abstractmethod

from ..retrieval import BaseRetriever, BM25Retriever, HybridRetriever


class BaseCommand(ABC):

    @abstractmethod
    def cmd(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    def get_retriever(mode: str, processed_dir: str) -> BaseRetriever:
        if mode not in ("bm25", "hybrid"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'bm25' or 'hybrid'."
            )
        return (
            HybridRetriever(processed_dir) if mode == "hybrid"
            else BM25Retriever(processed_dir)
        )
