from markdown_it.token import Token
from typing import List, Tuple, Optional

from ..base import Span
from ..line_map import LineMap

CODE_TYPES: Tuple[str, ...] = ("fence", "code_block")

TABLE_TYPE: str = "table_open"

HEADING_TYPE: str = "heading_open"


class MarkdownAstParser:

    def __init__(self, lmap: LineMap) -> None:
        self.__lmap: LineMap = lmap

    def extract_headings(
        self, tokens: List[Token],
    ) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for tok in tokens:
            if tok.type == HEADING_TYPE and tok.map:
                level: int = int(tok.tag[1:])
                offset: int = self.__lmap.line_start(
                    tok.map[0],
                )
                result.append((level, offset))
        return result

    def extract_blocks(
        self, tokens: List[Token],
    ) -> List[Tuple[str, Span]]:
        found: List[Tuple[str, Span]] = []
        for tok in tokens:
            bounds: Optional[List[int]] = tok.map
            if not bounds:
                continue
            start: int = self.__lmap.line_start(bounds[0])
            end: int = self.__lmap.line_start(bounds[1])
            if tok.type in CODE_TYPES:
                found.append(("code", (start, end)))
            elif tok.type == TABLE_TYPE:
                found.append(("table", (start, end)))
        return self._sanitize(found)

    def _sanitize(
        self, blocks: List[Tuple[str, Span]],
    ) -> List[Tuple[str, Span]]:
        ordered: List[Tuple[str, Span]] = sorted(
            blocks, key=lambda item: item[1][0],
        )
        clean: List[Tuple[str, Span]] = []
        pos: int = 0
        for kind, (s, e) in ordered:
            if s >= pos and s < e:
                clean.append((kind, (s, e)))
                pos = e
        return clean
