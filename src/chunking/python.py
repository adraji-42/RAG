import ast
from typing import List

from .base import BaseChunker, Span
from .line_map import LineMap
from ..models import MinimalSource


class AstSpanExtractor:

    def __init__(
        self, lmap: LineMap, max_size: int,
    ) -> None:
        self._lmap: LineMap = lmap
        self._max: int = max_size

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
                spans.extend(self._compound(node))
                gs, ge = -1, -1
            else:
                ns: int = self._offset_start(node)
                ne: int = self._offset_end(node)
                if gs < 0:
                    gs, ge = ns, ne
                else:
                    ge = ne
        if gs >= 0:
            spans.append((gs, ge))
        return spans

    def _offset_start(self, node: ast.stmt) -> int:
        return self._lmap.offset(
            node.lineno, node.col_offset,
        )

    def _offset_end(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self._lmap.offset(
            node.end_lineno, node.end_col_offset,
        )

    def _compound(self, node: ast.stmt) -> List[Span]:
        s: int = self._offset_start(node)
        e: int = self._offset_end(node)
        if isinstance(node, ast.ClassDef) and e - s > self._max:
            return self._split_cls(node)
        if not isinstance(node, ast.ClassDef) and e - s > self._max:
            return self._split_fn(node)
        return [(s, e)]

    def _split_cls(
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
            return [(self._offset_start(node), self._offset_end(node))]
        spans: List[Span] = []
        cs: int = self._offset_start(node)
        fm: int = self._offset_start(ms[0])
        if fm > cs:
            spans.append((cs, fm))
        cls_end: int = self._offset_end(node)
        for i, m in enumerate(ms):
            ms_start: int = self._offset_start(m)
            ms_end: int = (
                self._offset_start(ms[i + 1])
                if i + 1 < len(ms) else cls_end
            )
            if ms_end - ms_start > self._max:
                spans.extend(self._split_fn(m))
            else:
                spans.append((ms_start, ms_end))
        return spans

    def _split_fn(self, node: ast.stmt) -> List[Span]:
        body: List[ast.stmt] = getattr(
            node, "body", [],
        )
        if not body:
            return [(self._offset_start(node), self._offset_end(node))]
        spans: List[Span] = []
        fs: int = self._offset_start(node)
        bs: int = self._offset_start(body[0])
        if bs > fs:
            spans.append((fs, bs))
        gs: int = self._offset_start(body[0])
        ge: int = self._offset_end(body[0])
        for st in body[1:]:
            se: int = self._offset_end(st)
            if se - gs <= self._max:
                ge = se
            else:
                spans.append((gs, ge))
                gs = self._offset_start(st)
                ge = se
        spans.append((gs, max(ge, self._offset_end(node))))
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
            spans: List[Span] = self._split_paragraphs(
                content, 0, len(content),
            )
            return self._emit(file_path, content, spans)
        ext: AstSpanExtractor = AstSpanExtractor(
            lmap, self._max_chunk_size,
        )
        filled: List[Span] = self._fill_gaps(
            ext.extract(tree), len(content),
        )
        return self._emit(file_path, content, filled)

    def _fill_gaps(
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
