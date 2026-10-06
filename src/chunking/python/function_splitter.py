import ast
from typing import List

from ..base import Span
from ..line_map import LineMap


class FunctionSplitter:

    def __init__(self, lmap: LineMap, max_size: int) -> None:
        self.__lmap: LineMap = lmap
        self.__max: int = max_size

    def split(self, node: ast.stmt) -> List[Span]:
        body: List[ast.stmt] = getattr(node, "body", [])
        if not body:
            return [(self.__start(node), self.__end(node))]
        spans: List[Span] = []
        fs: int = self.__start(node)
        bs: int = self.__start(body[0])
        if bs > fs:
            spans.append((fs, bs))
        gs: int = self.__start(body[0])
        ge: int = self.__end(body[0])
        for st in body[1:]:
            se: int = self.__end(st)
            if se - gs <= self.__max:
                ge = se
            else:
                spans.append((gs, ge))
                gs = self.__start(st)
                ge = se
        spans.append((gs, max(ge, self.__end(node))))
        return spans

    def __start(self, node: ast.stmt) -> int:
        return self.__lmap.offset(node.lineno, node.col_offset)

    def __end(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.__lmap.offset(node.end_lineno, node.end_col_offset)
