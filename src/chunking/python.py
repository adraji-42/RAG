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
            gs, ge = self._visit(
                node, spans, gs, ge,
            )
        if gs >= 0:
            spans.append((gs, ge))
        return spans

    def _visit(
        self, node: ast.stmt, spans: List[Span],
        gs: int, ge: int,
    ) -> Span:
        if isinstance(node, (
            ast.ClassDef, ast.FunctionDef,
            ast.AsyncFunctionDef,
        )):
            if gs >= 0:
                spans.append((gs, ge))
            spans.extend(self._compound(node))
            return -1, -1
        ns: int = self._s(node)
        ne: int = self._e(node)
        return (ns, ne) if gs < 0 else (gs, ne)

    def _s(self, node: ast.stmt) -> int:
        return self._lmap.offset(
            node.lineno, node.col_offset,
        )

    def _e(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self._lmap.offset(
            node.end_lineno, node.end_col_offset,
        )

    def _compound(self, node: ast.stmt) -> List[Span]:
        s: int = self._s(node)
        e: int = self._e(node)
        if isinstance(node, ast.ClassDef):
            if e - s > self._max:
                return self._split_cls(node)
        elif e - s > self._max:
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
            return [(self._s(node), self._e(node))]
        return self._cls_parts(node, ms)

    def _cls_parts(
        self, node: ast.ClassDef,
        ms: List[ast.stmt],
    ) -> List[Span]:
        spans: List[Span] = []
        cs: int = self._s(node)
        fm: int = self._s(ms[0])
        if fm > cs:
            spans.append((cs, fm))
        spans.extend(
            self._meth_spans(ms, self._e(node)),
        )
        return spans

    def _meth_spans(
        self, ms: List[ast.stmt], cls_end: int,
    ) -> List[Span]:
        spans: List[Span] = []
        for i, m in enumerate(ms):
            s: int = self._s(m)
            e: int = (
                self._s(ms[i + 1])
                if i + 1 < len(ms) else cls_end
            )
            if e - s > self._max:
                spans.extend(self._split_fn(m))
            else:
                spans.append((s, e))
        return spans

    def _split_fn(self, node: ast.stmt) -> List[Span]:
        body: List[ast.stmt] = getattr(
            node, "body", [],
        )
        if not body:
            return [(self._s(node), self._e(node))]
        return self._fn_body(node, body)

    def _fn_body(
        self, node: ast.stmt,
        body: List[ast.stmt],
    ) -> List[Span]:
        spans: List[Span] = []
        fs: int = self._s(node)
        bs: int = self._s(body[0])
        if bs > fs:
            spans.append((fs, bs))
        spans.extend(
            self._greedy(body, self._e(node)),
        )
        return spans

    def _greedy(
        self, stmts: List[ast.stmt], end: int,
    ) -> List[Span]:
        spans: List[Span] = []
        gs: int = self._s(stmts[0])
        ge: int = self._e(stmts[0])
        for st in stmts[1:]:
            se: int = self._e(st)
            if se - gs <= self._max:
                ge = se
            else:
                spans.append((gs, ge))
                gs, ge = self._s(st), se
        spans.append((gs, max(ge, end)))
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
            return self._fallback(file_path, content)
        return self._from_ast(
            file_path, content, lmap, tree,
        )

    def _from_ast(
        self, fp: str, content: str,
        lmap: LineMap, tree: ast.Module,
    ) -> List[MinimalSource]:
        ext: AstSpanExtractor = AstSpanExtractor(
            lmap, self._max_chunk_size,
        )
        filled: List[Span] = self._fill(
            ext.extract(tree), len(content),
        )
        return self._packer.emit(fp, content, filled)

    def _fallback(
        self, fp: str, content: str,
    ) -> List[MinimalSource]:
        spans: List[Span] = (
            self._packer.split_paragraphs(
                content, 0, len(content),
            )
        )
        return self._packer.emit(fp, content, spans)

    def _fill(
        self, spans: List[Span], total: int,
    ) -> List[Span]:
        if not spans:
            return [(0, total)] if total > 0 else []
        filled: List[Span] = []
        if spans[0][0] > 0:
            filled.append((0, spans[0][0]))
        filled.extend(self._inter(spans))
        if spans[-1][1] < total:
            filled.append((spans[-1][1], total))
        return filled

    def _inter(self, spans: List[Span]) -> List[Span]:
        filled: List[Span] = []
        prev: int = spans[0][0]
        for s, e in spans:
            if s > prev:
                filled.append((prev, s))
            filled.append((s, e))
            prev = e
        return filled
