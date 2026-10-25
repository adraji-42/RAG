from typing import Any
from abc import ABC, abstractmethod


class BaseCommand(ABC):

    @abstractmethod
    def cmd(self, *args: Any, **kwargs: Any) -> Any:
        pass
