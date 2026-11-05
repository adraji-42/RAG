"""Class statement boundary splitting for oversized Python classes."""

import ast
from typing import List

from ..base import Span
from ..line_map import LineMap
from .function_splitter import FunctionSplitter


class ClassSplitter:
    """Split oversized Python class definitions into member-level spans."""

    def __init__(
        self, lmap: LineMap, fn_splitter: FunctionSplitter, max_size: int,
    ) -> None:
        """Initialize class splitter with line map and function splitter.

        Args:
            lmap: LineMap instance for line and offset mapping.
            fn_splitter: FunctionSplitter instance for method splitting.
            max_size: Maximum allowed characters per chunk.
        """
        self.__lmap: LineMap = lmap
        self.__fn_splitter: FunctionSplitter = fn_splitter
        self.__max: int = max_size

    def split(self, node: ast.ClassDef, begin: int = -1) -> List[Span]:
        """Split an oversized class AST node into smaller statement spans.

        Args:
            node: Parsed Python ClassDef AST node.
            begin: Optional starting offset override.

        Returns:
            List of character span tuples for split class members.
        """
        s: int = begin if begin >= 0 else self.__start(node)
        e: int = self.__end(node)
        if e - s <= self.__max:
            return [(s, e)]

        cs: int = s
        ce: int = -1
        prev: int = s
        spans: List[Span] = []

        for st in node.body:
            se: int = self.__end(st)
            if cs < 0:
                cs = self.__resume(prev, st)
            if ce >= 0 and se - cs > self.__max:
                spans.append((cs, ce))
                cs = self.__resume(prev, st)
                ce = -1
            if se - cs > self.__max:
                spans.extend(self.__oversized(st, cs))
                cs = -1
            else:
                ce = se
            prev = se
        if ce >= 0:
            spans.append((cs, e))
        return spans

    def __oversized(self, node: ast.stmt, begin: int) -> List[Span]:
        """Handle an individual oversized member node inside a class.

        Args:
            node: Oversized member AST statement.
            begin: Starting character offset for the statement.

        Returns:
            List of character span tuples for the oversized member.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.__fn_splitter.split(node, begin)
        if isinstance(node, ast.ClassDef):
            return self.split(node, begin)
        return [(begin, self.__end(node))]

    def __resume(self, prev_end: int, node: ast.stmt) -> int:
        """Compute resuming character offset after a statement boundary.

        Args:
            prev_end: End character offset of previous statement.
            node: Current AST statement node.

        Returns:
            Resuming character offset aligned to line boundaries.
        """
        line: int = self.__lmap.line_at(prev_end) + 1
        return min(self.__lmap.line_start(line), self.__start(node))

    def __start(self, node: ast.stmt) -> int:
        """Return start character offset of an AST node.

        Args:
            node: AST statement node.

        Returns:
            Start character offset in the source document.
        """
        return self.__lmap.offset(node.lineno, node.col_offset)

    def __end(self, node: ast.stmt) -> int:
        """Return end character offset of an AST node.

        Args:
            node: AST statement node with end positions.

        Returns:
            End character offset in the source document.
        """
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.__lmap.offset(node.end_lineno, node.end_col_offset)
