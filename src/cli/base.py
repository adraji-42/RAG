"""Abstract base class and retriever factory for CLI subcommands."""

from typing import Any
from abc import ABC, abstractmethod

from ..retrieval import BaseRetriever, BM25Retriever, HybridRetriever


class BaseCommand(ABC):
    """Abstract base command for CLI subcommands."""

    @abstractmethod
    def cmd(self, *args: Any, **kwargs: Any) -> Any:
        """Execute command logic with arbitrary arguments.

        Args:
            *args: Positional command arguments.
            **kwargs: Keyword command arguments.

        Returns:
            Result of command execution.
        """
        pass

    @staticmethod
    def get_retriever(mode: str, processed_dir: str) -> BaseRetriever:
        """Return configured BM25 or Hybrid retriever based on mode string.

        Args:
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
            processed_dir: Path to directory containing index files.

        Returns:
            Configured BaseRetriever instance.

        Raises:
            ValueError: If mode is not 'bm25' or 'hybrid'.
        """
        if mode not in ("bm25", "hybrid"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'bm25' or 'hybrid'."
            )
        return (
            HybridRetriever(processed_dir) if mode == "hybrid"
            else BM25Retriever(processed_dir)
        )
