"""Line offset mapping table for character index translations."""

from typing import List
from bisect import bisect_right


class LineMap:
    """Map 1-indexed line numbers and columns to character offsets."""

    def __init__(self, content: str) -> None:
        """Build newline byte offset table for fast character lookups.

        Args:
            content: Document text content string.
        """
        self.__offsets: List[int] = [0]
        pos: int = content.find("\n")
        while pos != -1:
            self.__offsets.append(pos + 1)
            pos = content.find("\n", pos + 1)
        self.__total: int = len(content)

    @property
    def offsets(self) -> List[int]:
        """Return the list of line starting character offsets.

        Returns:
            List of integer character offsets for line starts.
        """
        return self.__offsets

    @property
    def total(self) -> int:
        """Return the total character length of the content.

        Returns:
            Total character count of the document content.
        """
        return self.__total

    def offset(self, line: int, col: int = 0) -> int:
        """Compute character offset from 1-indexed line and column.

        Args:
            line: 1-indexed line number.
            col: 0-indexed column offset.

        Returns:
            Absolute 0-indexed character offset.
        """
        return self.__offsets[line - 1] + col

    def line_at(self, offset: int) -> int:
        """Return 0-indexed line containing the given character offset.

        Args:
            offset: 0-indexed character offset in document.

        Returns:
            0-indexed line index containing the offset.
        """
        return bisect_right(self.__offsets, offset) - 1

    def line_start(self, line_idx: int) -> int:
        """Return start offset of the given 0-indexed line.

        Args:
            line_idx: 0-indexed line index.

        Returns:
            Start character offset for the specified line.
        """
        if line_idx < len(self.__offsets):
            return self.__offsets[line_idx]
        return self.__total
