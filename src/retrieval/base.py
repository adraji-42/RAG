from abc import ABC, abstractmethod
from typing import List

from ..models import MinimalSource


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        pass
