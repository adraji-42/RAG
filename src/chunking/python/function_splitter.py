import ast
from typing import List

from ..base import Span
from ..line_map import LineMap


class FunctionSplitter:

    def __init__(self, lmap: LineMap, max_size: int) -> None:
        self.__lmap: LineMap = lmap
        self.__max: int = max_size

    def split(self, node: ast.stmt, begin: int = -1) -> List[Span]:
        body: List[ast.stmt] = getattr(node, "body", [])
        fs: int = begin if begin >= 0 else self.__start(node)
        end: int = self.__end(node)
        if not body:
            return [(fs, end)]

        cs: int = fs
        ce: int = -1
        prev: int = fs
        spans: List[Span] = []

        for st in body:
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
            spans.append((cs, end))
        return spans

    def __oversized(self, node: ast.stmt, begin: int) -> List[Span]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.split(node, begin)
        if isinstance(node, ast.ClassDef):
            from .class_splitter import ClassSplitter
            return ClassSplitter(
                self.__lmap, self, self.__max
            ).split(node, begin)
        return [(begin, self.__end(node))]

    def __resume(self, prev_end: int, node: ast.stmt) -> int:
        line: int = self.__lmap.line_at(prev_end) + 1
        return min(self.__lmap.line_start(line), self.__start(node))

    def __start(self, node: ast.stmt) -> int:
        return self.__lmap.offset(node.lineno, node.col_offset)

    def __end(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.__lmap.offset(node.end_lineno, node.end_col_offset)
