from typing import List, Tuple
from markdown_it import MarkdownIt
from markdown_it.token import Token

from ..line_map import LineMap
from ..text import TextChunker
from ...models import MinimalSource
from ..base import BaseChunker, Span
from .ast_parser import MarkdownAstParser
from .block_isolator import BlockIsolator
from .heading_splitter import HeadingSplitter


class MarkdownChunker(BaseChunker):

    def __init__(
        self, max_chunk_size: int = 2000,
    ) -> None:
        super().__init__(max_chunk_size)
        self.__md: MarkdownIt = MarkdownIt("gfm-like").disable("linkify")
        self.__text: TextChunker = TextChunker(max_chunk_size)
        self.__isolator: BlockIsolator = BlockIsolator(
            self.__text, max_chunk_size,
        )
        self.__splitter: HeadingSplitter = HeadingSplitter(
            self.__isolator, max_chunk_size,
        )

    def chunk(
        self, file_path: str, content: str,
    ) -> List[MinimalSource]:
        lmap: LineMap = LineMap(content)
        tokens: List[Token] = self.__md.parse(content)
        parser: MarkdownAstParser = MarkdownAstParser(lmap)
        heads: List[Tuple[int, int]] = parser.extract_headings(tokens)
        blocks: List[Tuple[str, Span]] = parser.extract_blocks(tokens)
        spans: List[Span] = self.__splitter.split(
            content, 0, len(content), 0, heads, blocks,
        )
        return self.emit(file_path, content, spans)
