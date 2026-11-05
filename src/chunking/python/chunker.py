"""AST-guided syntax-aware chunker for Python source code files."""

import ast
from typing import List

from ..line_map import LineMap
from ..text import TextChunker
from ...models import MinimalSource
from ..base import BaseChunker, Span
from .class_splitter import ClassSplitter
from .ast_extractor import AstSpanExtractor
from .function_splitter import FunctionSplitter


class PythonChunker(BaseChunker):
    """Chunk Python source files using AST statement extraction."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize Python chunker with a maximum character size.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
        """
        super().__init__(max_chunk_size)

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        """Parse Python code into AST spans and emit minimal source chunks.

        Args:
            file_path: Source Python file path string.
            content: Python source code content.

        Returns:
            List of MinimalSource chunks representing syntax-aware spans.
        """
        try:
            tree: ast.Module = ast.parse(content)
        except SyntaxError:
            chunker: TextChunker = TextChunker(self.max_chunk_size)
            return chunker.chunk(file_path, content)

        lmap: LineMap = LineMap(content)
        fn_splitter: FunctionSplitter = FunctionSplitter(
            lmap, self.max_chunk_size
        )
        cls_splitter: ClassSplitter = ClassSplitter(
            lmap, fn_splitter, self.max_chunk_size
        )
        ext: AstSpanExtractor = AstSpanExtractor(
            lmap, cls_splitter, fn_splitter, self.max_chunk_size
        )
        filled: List[Span] = self.__fill_gaps(ext.extract(tree), len(content))

        return self.emit(file_path, content, filled)

    def __fill_gaps(
        self, spans: List[Span], total: int,
    ) -> List[Span]:
        """Fill uncaptured character gaps between extracted AST spans.

        Args:
            spans: List of extracted non-contiguous AST spans.
            total: Total character length of the file content.

        Returns:
            List of complete spans filling gaps between AST statements.
        """
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
