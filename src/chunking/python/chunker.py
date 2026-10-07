import ast
from typing import List

from ..base import BaseChunker, Span
from ..line_map import LineMap
from ...models import MinimalSource
from .ast_extractor import AstSpanExtractor
from .class_splitter import ClassSplitter
from .function_splitter import FunctionSplitter


class PythonChunker(BaseChunker):

    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        try:
            tree: ast.Module = ast.parse(content)
        except SyntaxError:
            spans: List[Span] = self.split_paragraphs(content, 0, len(content))
            return self.emit(file_path, content, spans)

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
