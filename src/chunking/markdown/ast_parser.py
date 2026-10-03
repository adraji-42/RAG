from typing import Dict, List, Tuple

from markdown_it.token import Token

from ..base import Span
from ..line_map import LineMap

BLOCK_OPENERS: Dict[str, str] = {
    "table_open": "table_close",
    "bullet_list_open": "bullet_list_close",
    "ordered_list_open": "ordered_list_close",
    "blockquote_open": "blockquote_close",
}

FENCE_TYPES: Tuple[str, ...] = ("fence", "code_block")

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
        result: List[Tuple[str, Span]] = []
        i: int = 0
        while i < len(tokens):
            tok: Token = tokens[i]
            if tok.type in FENCE_TYPES and tok.map:
                s: int = self.__lmap.line_start(tok.map[0])
                e: int = self.__lmap.line_start(tok.map[1])
                result.append(("code", (s, e)))
            elif tok.type in BLOCK_OPENERS and tok.map:
                closer: str = BLOCK_OPENERS[tok.type]
                bs: int = self.__lmap.line_start(tok.map[0])
                be: int = self._find_close(tokens, i, closer)
                result.append(("block", (bs, be)))
            i += 1
        return result

    def _find_close(
        self, tokens: List[Token],
        start_idx: int, closer: str,
    ) -> int:
        depth: int = 1
        opener: str = tokens[start_idx].type
        for j in range(start_idx + 1, len(tokens)):
            if tokens[j].type == opener:
                depth += 1
            elif tokens[j].type == closer:
                depth -= 1
                cmap: List[int] | None = tokens[j].map
                if depth == 0 and cmap:
                    return self.__lmap.line_start(cmap[1])
                if depth == 0:
                    return self._fallback_close(
                        tokens, start_idx,
                    )
        return self._fallback_close(tokens, start_idx)

    def _fallback_close(
        self, tokens: List[Token], start_idx: int,
    ) -> int:
        m: object = tokens[start_idx].map
        if m and isinstance(m, list) and len(m) >= 2:
            return self.__lmap.line_start(m[1])
        return self.__lmap.total
