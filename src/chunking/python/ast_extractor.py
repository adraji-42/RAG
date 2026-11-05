"""AST statement extraction and compound statement span processing."""

import ast
from typing import List

from ..base import Span
from ..line_map import LineMap
from .class_splitter import ClassSplitter
from .function_splitter import FunctionSplitter

_COMPOUND = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


class AstSpanExtractor:
    """Extract character spans for AST top-level and compound nodes."""

    def __init__(
        self,
        lmap: LineMap,
        cls_splitter: ClassSplitter,
        fn_splitter: FunctionSplitter,
        max_size: int,
    ) -> None:
        """Initialize the AST span extractor with line map and splitters.

        Args:
            lmap: LineMap instance for offset lookups.
            cls_splitter: ClassSplitter instance for splitting classes.
            fn_splitter: FunctionSplitter instance for splitting functions.
            max_size: Maximum allowed characters per chunk.
        """
        self.__lmap: LineMap = lmap
        self.__cls_splitter: ClassSplitter = cls_splitter
        self.__fn_splitter: FunctionSplitter = fn_splitter
        self.__max: int = max_size

    def extract(self, tree: ast.Module) -> List[Span]:
        """Extract character spans from top-level module statements.

        Args:
            tree: Parsed Python AST module node.

        Returns:
            List of character span tuples covering module statements.
        """
        gs: int = -1
        ge: int = -1
        spans: List[Span] = []
        for node in tree.body:
            if isinstance(node, _COMPOUND):
                if gs >= 0:
                    spans.append((gs, ge))
                    gs = -1
                spans.extend(self.__compound(node))
                continue
            ne: int = self.__end(node)
            if gs >= 0 and ne - gs > self.__max:
                spans.append((gs, ge))
                gs = -1
            if gs < 0:
                gs = self.__start(node)
            ge = ne
        if gs >= 0:
            spans.append((gs, ge))
        return spans

    def __compound(self, node: ast.stmt) -> List[Span]:
        """Process and split a compound AST statement node.

        Args:
            node: AST class or function definition statement.

        Returns:
            List of character span tuples for the compound statement.
        """
        s: int = self.__start(node)
        e: int = self.__end(node)
        if e - s <= self.__max:
            return [(s, e)]
        if isinstance(node, ast.ClassDef):
            return self.__cls_splitter.split(node)
        return self.__fn_splitter.split(node)

    def __start(self, node: ast.stmt) -> int:
        """Return start character offset of an AST statement.

        Args:
            node: AST statement node.

        Returns:
            Start character offset in the source document.
        """
        return self.__lmap.offset(node.lineno, node.col_offset)

    def __end(self, node: ast.stmt) -> int:
        """Return end character offset of an AST statement.

        Args:
            node: AST statement node with end positions.

        Returns:
            End character offset in the source document.
        """
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.__lmap.offset(node.end_lineno, node.end_col_offset)
