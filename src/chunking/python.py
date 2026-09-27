"""AST-based Python file chunker implementing the BaseChunker strategy."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Union

from .base import BaseChunker, Chunk


_STRUCTURAL_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_StructuralNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


class PythonChunker(BaseChunker):
    """Chunks Python source by function and class definitions.

    Each definition becomes its own chunk, kept together with any
    decorators it carries. A class that would still exceed
    ``max_chunk_size`` is recursively split method by method instead
    of being cut at an arbitrary character offset.
    """

    def chunk(self, file_path: Path, content: str) -> Iterator[Chunk]:
        """Yield chunks for *content* parsed as a Python source file."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            yield from self._split_by_size(content, 0, str(file_path))
            return

        line_offsets = self._build_line_offsets(content)
        yield from self._emit_body(
            content, tree.body, 0, len(content), line_offsets, str(file_path),
        )

    @staticmethod
    def _build_line_offsets(source: str) -> list[int]:
        """Return char offset of the start of each line (0-indexed list)."""
        offsets: list[int] = [0]
        for i, ch in enumerate(source):
            if ch == "\n" and i + 1 < len(source):
                offsets.append(i + 1)
        return offsets

    @staticmethod
    def _node_span(
        source: str,
        node: _StructuralNode,
        line_offsets: list[int],
    ) -> tuple[int, str]:
        """Return ``(start_char, text)`` for *node*, decorators included.

        ``node.lineno`` points at the ``def``/``class`` keyword, not at
        any decorator above it, so a decorated definition would
        otherwise be split from its decorators. Starting from the
        first decorator's line (when present) keeps them together.
        """
        start_line = (
            node.decorator_list[0].lineno
            if node.decorator_list
            else node.lineno
        )
        start = line_offsets[start_line - 1]
        end_lineno = (
            node.end_lineno if node.end_lineno is not None else start_line
        )
        end = (
            line_offsets[end_lineno]
            if end_lineno < len(line_offsets)
            else len(source)
        )
        return start, source[start:end]

    def _emit_node(
        self,
        source: str,
        node: _StructuralNode,
        start: int,
        text: str,
        line_offsets: list[int],
        file_path: str,
    ) -> Iterator[Chunk]:
        """Yield one or more size-capped chunks for a single AST *node*."""
        if len(text) <= self.max_chunk_size:
            yield Chunk.spanning(file_path, start, text)
            return
        if isinstance(node, ast.ClassDef):
            yield from self._emit_body(
                source, node.body, start, start + len(text),
                line_offsets, file_path,
            )
        else:
            yield from self._split_by_size(text, start, file_path)

    def _emit_gap(
        self,
        source: str,
        gap_start: int,
        gap_end: int,
        file_path: str,
    ) -> Iterator[Chunk]:
        """Yield chunks for the gap of source text between two nodes."""
        gap = source[gap_start:gap_end]
        if gap.strip():
            yield from self._split_by_size(gap, gap_start, file_path)

    def _emit_body(
        self,
        source: str,
        body: list[ast.stmt],
        region_start: int,
        region_end: int,
        line_offsets: list[int],
        file_path: str,
    ) -> Iterator[Chunk]:
        """Walk one AST body (a module or a class), in document order.

        Used both for the module's top level and, recursively, for the
        body of a class that turned out too large to keep as one chunk
        -- so an oversized class is split along its methods rather than
        at a character offset that could land mid-statement.
        """
        covered_up_to = region_start
        for node in body:
            if not isinstance(node, _STRUCTURAL_NODES):
                continue
            start, text = self._node_span(source, node, line_offsets)
            yield from self._emit_gap(source, covered_up_to, start, file_path)
            yield from self._emit_node(
                source, node, start, text, line_offsets, file_path,
            )
            covered_up_to = start + len(text)
        if covered_up_to < region_end:
            yield from self._emit_gap(
                source, covered_up_to, region_end, file_path,
            )
