import ast
from typing import List

from .base import BaseChunker, Span
from .line_map import LineMap
from ..models import MinimalSource


class AstSpanExtractor:

    def __init__(
        self, lmap: LineMap, max_size: int,
    ) -> None:
        self.__lmap: LineMap = lmap
        self.__max: int = max_size

    @property
    def lmap(self) -> LineMap:
        return self.__lmap

    @property
    def max_size(self) -> int:
        return self.__max

    def extract(self, tree: ast.Module) -> List[Span]:
        spans: List[Span] = []
        gs: int = -1
        ge: int = -1
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.stmt):
                continue
            if isinstance(node, (
                ast.ClassDef, ast.FunctionDef,
                ast.AsyncFunctionDef,
            )):
                if gs >= 0:
                    spans.append((gs, ge))
                spans.extend(self.__compound(node))
                gs, ge = -1, -1
            else:
                ns: int = self.__offset_start(node)
                ne: int = self.__offset_end(node)
                if gs < 0:
                    gs, ge = ns, ne
                else:
                    ge = ne
        if gs >= 0:
            spans.append((gs, ge))
        return spans

    def __offset_start(self, node: ast.stmt) -> int:
        return self.__lmap.offset(
            node.lineno, node.col_offset,
        )

    def __offset_end(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.__lmap.offset(
            node.end_lineno, node.end_col_offset,
        )

    def __compound(self, node: ast.stmt) -> List[Span]:
        s: int = self.__offset_start(node)
        e: int = self.__offset_end(node)
        if isinstance(node, ast.ClassDef) and e - s > self.__max:
            return self.__split_cls(node)
        if not isinstance(node, ast.ClassDef) and e - s > self.__max:
            return self.__split_fn(node)
        return [(s, e)]

    def __split_cls(
        self, node: ast.ClassDef,
    ) -> List[Span]:
        ms: List[ast.stmt] = [
            c for c in ast.iter_child_nodes(node)
            if isinstance(c, ast.stmt)
            and isinstance(c, (
                ast.FunctionDef, ast.AsyncFunctionDef,
            ))
        ]
        if not ms:
            return [(self.__offset_start(node), self.__offset_end(node))]
        spans: List[Span] = []
        cs: int = self.__offset_start(node)
        fm: int = self.__offset_start(ms[0])
        if fm > cs:
            spans.append((cs, fm))
        cls_end: int = self.__offset_end(node)
        for i, m in enumerate(ms):
            ms_start: int = self.__offset_start(m)
            ms_end: int = (
                self.__offset_start(ms[i + 1])
                if i + 1 < len(ms) else cls_end
            )
            if ms_end - ms_start > self.__max:
                spans.extend(self.__split_fn(m))
            else:
                spans.append((ms_start, ms_end))
        return spans

    def __split_fn(self, node: ast.stmt) -> List[Span]:
        body: List[ast.stmt] = getattr(
            node, "body", [],
        )
        if not body:
            return [(self.__offset_start(node), self.__offset_end(node))]
        spans: List[Span] = []
        fs: int = self.__offset_start(node)
        bs: int = self.__offset_start(body[0])
        if bs > fs:
            spans.append((fs, bs))
        gs: int = self.__offset_start(body[0])
        ge: int = self.__offset_end(body[0])
        for st in body[1:]:
            se: int = self.__offset_end(st)
            if se - gs <= self.__max:
                ge = se
            else:
                spans.append((gs, ge))
                gs = self.__offset_start(st)
                ge = se
        spans.append((gs, max(ge, self.__offset_end(node))))
        return spans


class PythonChunker(BaseChunker):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        super().__init__(max_chunk_size)

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        lmap: LineMap = LineMap(content)
        try:
            tree: ast.Module = ast.parse(content)
        except SyntaxError:
            spans: List[Span] = self.split_paragraphs(
                content, 0, len(content),
            )
            return self.emit(file_path, content, spans)
        ext: AstSpanExtractor = AstSpanExtractor(
            lmap, self.max_chunk_size,
        )
        filled: List[Span] = self.__fill_gaps(
            ext.extract(tree), len(content),
        )
        return self.emit(file_path, content, filled)

    def __fill_gaps(
        self, spans: List[Span], total: int,
    ) -> List[Span]:
        if not spans:
            return [(0, total)] if total > 0 else []
        filled: List[Span] = []
        if spans[0][0] > 0:
            filled.append((0, spans[0][0]))
        prev: int = spans[0][0]
        for s, e in spans:
            if s > prev:
                filled.append((prev, s))
            filled.append((s, e))
            prev = e
        if spans[-1][1] < total:
            filled.append((spans[-1][1], total))
        return filled
