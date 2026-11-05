"""Abstract base class for document retrieval implementations."""

from typing import List
from abc import ABC, abstractmethod

from ..models import MinimalSource


class BaseRetriever(ABC):
    """Abstract base class for document retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Retrieve top-k relevant source spans matching the query.

        Args:
            query: Search query text string.
            k: Maximum number of source spans to return.

        Returns:
            List of retrieved MinimalSource spans.
        """
        pass
