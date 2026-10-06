import ast
from typing import List

from ..base import Span
from ..line_map import LineMap
from .function_splitter import FunctionSplitter


class ClassSplitter:

    def __init__(
        self, lmap: LineMap, fn_splitter: FunctionSplitter, max_size: int,
    ) -> None:
        self.__lmap: LineMap = lmap
        self.__fn_splitter: FunctionSplitter = fn_splitter
        self.__max: int = max_size

    def split(self, node: ast.ClassDef) -> List[Span]:
        s: int = self.__start(node)
        e: int = self.__end(node)
        if e - s <= self.__max:
            return [(s, e)]

        ms: List[ast.stmt] = [
            c for c in ast.iter_child_nodes(node)
            if isinstance(c, ast.stmt)
            and isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not ms:
            return [(s, e)]

        spans: List[Span] = []
        cs: int = s
        fm: int = self.__start(ms[0])
        if fm > cs:
            spans.append((cs, fm))
        cls_end: int = e
        for i, m in enumerate(ms):
            ms_start: int = self.__start(m)
            ms_end: int = (
                self.__start(ms[i + 1])
                if i + 1 < len(ms) else cls_end
            )
            if ms_end - ms_start > self.__max:
                spans.extend(self.__fn_splitter.split(m))
            else:
                spans.append((ms_start, ms_end))
        return spans

    def __start(self, node: ast.stmt) -> int:
        return self.__lmap.offset(node.lineno, node.col_offset)

    def __end(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.__lmap.offset(node.end_lineno, node.end_col_offset)
