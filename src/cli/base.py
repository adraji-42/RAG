from abc import ABC, abstractmethod
from typing import Any


class BaseCommand(ABC):

    @abstractmethod
    def cmd(self, *args: Any, **kwargs: Any) -> Any:
        pass
