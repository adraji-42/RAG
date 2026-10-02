from bisect import bisect_right
from typing import List


class LineMap:

    def __init__(self, content: str) -> None:
        self.__offsets: List[int] = [0]
        pos: int = content.find("\n")
        while pos != -1:
            self.__offsets.append(pos + 1)
            pos = content.find("\n", pos + 1)
        self.__total: int = len(content)

    @property
    def offsets(self) -> List[int]:
        return self.__offsets

    @property
    def total(self) -> int:
        return self.__total

    def offset(self, line: int, col: int = 0) -> int:
        return self.__offsets[line - 1] + col

    def line_at(self, offset: int) -> int:
        return bisect_right(self.__offsets, offset) - 1

    def line_start(self, line_idx: int) -> int:
        if line_idx < len(self.__offsets):
            return self.__offsets[line_idx]
        return self.__total
