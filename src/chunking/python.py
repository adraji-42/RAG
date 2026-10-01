import ast

from typing import List, Tuple

from .base import BaseChunker
from ..models import MinimalSource


class PythonChunker(BaseChunker):

    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)
        self._content: str = ""
        self._line_offsets: List[int] = []

    def chunk(
        self,
        file_path: str,
        content: str,
    ) -> List[MinimalSource]:
        self._content = content
        self._line_offsets = self._build_line_offsets(
            content,
        )
        try:
            tree: ast.Module = ast.parse(content)
        except SyntaxError:
            return self._line_fallback(
                file_path, content,
            )
        spans: List[Tuple[int, int]] = (
            self._extract_spans(tree)
        )
        filled: List[Tuple[int, int]] = (
            self._fill_gaps(spans)
        )
        return self._spans_to_chunks(
            file_path, content, filled,
        )

    def _line_fallback(
        self,
        file_path: str,
        content: str,
    ) -> List[MinimalSource]:
        spans: List[Tuple[int, int]] = (
            self._paragraph_spans(content)
        )
        return self._spans_to_chunks(
            file_path, content, spans,
        )

    def _paragraph_spans(
        self,
        content: str,
    ) -> List[Tuple[int, int]]:
        parts: List[str] = content.split("\n\n")
        spans: List[Tuple[int, int]] = []
        pos: int = 0
        total: int = len(content)
        for part in parts:
            end: int = pos + len(part)
            if end < total:
                end += 2
            end = min(end, total)
            spans.append((pos, end))
            pos = end
        return spans

    def _line_col_to_offset(
        self,
        line: int,
        col: int,
    ) -> int:
        return self._line_offsets[line - 1] + col

    def _node_start(self, node: ast.stmt) -> int:
        return self._line_col_to_offset(
            node.lineno,
            node.col_offset,
        )

    def _node_end(self, node: ast.stmt) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self._line_col_to_offset(
            node.end_lineno,
            node.end_col_offset,
        )

    def _extract_spans(
        self,
        tree: ast.Module,
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        group_start: int = -1
        group_end: int = -1
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.stmt):
                continue
            if self._is_compound(node):
                group_start, group_end = (
                    self._flush_group(
                        spans, group_start, group_end,
                    )
                )
                spans.extend(self._node_spans(node))
            else:
                ns: int = self._node_start(node)
                ne: int = self._node_end(node)
                if group_start < 0:
                    group_start, group_end = ns, ne
                else:
                    group_end = ne
        self._flush_group(
            spans, group_start, group_end,
        )
        return spans

    def _is_compound(self, node: ast.stmt) -> bool:
        return isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )

    def _flush_group(
        self,
        spans: List[Tuple[int, int]],
        start: int,
        end: int,
    ) -> Tuple[int, int]:
        if start >= 0:
            spans.append((start, end))
        return -1, -1

    def _node_spans(
        self,
        node: ast.stmt,
    ) -> List[Tuple[int, int]]:
        start: int = self._node_start(node)
        end: int = self._node_end(node)
        size: int = end - start
        if isinstance(node, ast.ClassDef):
            return self._class_spans(node, size)
        if self._is_func(node) and size > self._max_chunk_size:
            return self._decompose_func(node)
        return [(start, end)]

    def _is_func(self, node: ast.stmt) -> bool:
        return isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef),
        )

    def _class_spans(
        self,
        node: ast.ClassDef,
        size: int,
    ) -> List[Tuple[int, int]]:
        if size <= self._max_chunk_size:
            return [(
                self._node_start(node),
                self._node_end(node),
            )]
        return self._decompose_class(node)

    def _decompose_class(
        self,
        node: ast.ClassDef,
    ) -> List[Tuple[int, int]]:
        methods: List[ast.stmt] = [
            c for c in ast.iter_child_nodes(node)
            if isinstance(c, ast.stmt) and self._is_func(c)
        ]
        if not methods:
            return [(
                self._node_start(node),
                self._node_end(node),
            )]
        return self._split_class(node, methods)

    def _split_class(
        self,
        node: ast.ClassDef,
        methods: List[ast.stmt],
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        cls_start: int = self._node_start(node)
        cls_end: int = self._node_end(node)
        first_m: int = self._node_start(methods[0])
        if first_m > cls_start:
            spans.append((cls_start, first_m))
        spans.extend(
            self._method_spans(methods, cls_end),
        )
        return spans

    def _method_spans(
        self,
        methods: List[ast.stmt],
        cls_end: int,
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for i, method in enumerate(methods):
            m_start: int = self._node_start(method)
            m_end: int = self._method_end(
                methods, i, cls_end,
            )
            if self._is_func(method):
                spans.extend(
                    self._maybe_decompose_func(
                        method, m_start, m_end,
                    )
                )
            else:
                spans.append((m_start, m_end))
        return spans

    def _method_end(
        self,
        methods: List[ast.stmt],
        i: int,
        cls_end: int,
    ) -> int:
        if i + 1 < len(methods):
            return self._node_start(methods[i + 1])
        return cls_end

    def _maybe_decompose_func(
        self,
        node: ast.stmt,
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        if end - start <= self._max_chunk_size:
            return [(start, end)]
        return self._decompose_func(node)

    def _decompose_func(
        self,
        node: ast.stmt,
    ) -> List[Tuple[int, int]]:
        body: List[ast.stmt] = getattr(node, "body", [])
        if not body:
            return [(
                self._node_start(node),
                self._node_end(node),
            )]
        return self._split_func_body(node, body)

    def _split_func_body(
        self,
        node: ast.stmt,
        body: List[ast.stmt],
    ) -> List[Tuple[int, int]]:
        func_start: int = self._node_start(node)
        func_end: int = self._node_end(node)
        first_stmt: int = self._node_start(body[0])
        spans: List[Tuple[int, int]] = []
        if first_stmt > func_start:
            spans.append((func_start, first_stmt))
        spans.extend(self._greedy_group(body, func_end))
        return spans

    def _greedy_group(
        self,
        stmts: List[ast.stmt],
        container_end: int,
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        g_start: int = self._node_start(stmts[0])
        g_end: int = self._node_end(stmts[0])
        for stmt in stmts[1:]:
            s_end: int = self._node_end(stmt)
            if s_end - g_start <= self._max_chunk_size:
                g_end = s_end
            else:
                spans.append((g_start, g_end))
                g_start = self._node_start(stmt)
                g_end = s_end
        spans.append((g_start, max(g_end, container_end)))
        return spans

    def _fill_gaps(
        self,
        spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not spans:
            return self._handle_empty()
        filled: List[Tuple[int, int]] = []
        if spans[0][0] > 0:
            filled.append((0, spans[0][0]))
        filled.extend(self._fill_inter_gaps(spans))
        total: int = len(self._content)
        if spans[-1][1] < total:
            filled.append((spans[-1][1], total))
        return filled

    def _handle_empty(
        self,
    ) -> List[Tuple[int, int]]:
        total: int = len(self._content)
        if total > 0:
            return [(0, total)]
        return []

    def _fill_inter_gaps(
        self,
        spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        filled: List[Tuple[int, int]] = []
        prev_end: int = spans[0][0]
        for start, end in spans:
            if start > prev_end:
                filled.append((prev_end, start))
            filled.append((start, end))
            prev_end = end
        return filled
