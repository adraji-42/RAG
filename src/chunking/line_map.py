from bisect import bisect_right
from typing import List


class LineMap:

    def __init__(self, content: str) -> None:
        self._offsets: List[int] = [0]
        for i, ch in enumerate(content):
            if ch == "\n":
                self._offsets.append(i + 1)
        self._total: int = len(content)

    def offset(self, line: int, col: int = 0) -> int:
        return self._offsets[line - 1] + col

    def line_at(self, offset: int) -> int:
        return bisect_right(self._offsets, offset) - 1

    def line_start(self, line_idx: int) -> int:
        if line_idx < len(self._offsets):
            return self._offsets[line_idx]
        return self._total

    @property
    def total(self) -> int:
        return self._total
