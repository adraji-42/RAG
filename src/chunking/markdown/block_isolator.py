from typing import List, Tuple

from ..base import Span
from ..text import TextChunker


class BlockIsolator:

    def __init__(
        self, text_chunker: TextChunker,
        max_chunk_size: int,
    ) -> None:
        self.__text: TextChunker = text_chunker
        self.__max: int = max_chunk_size

    def isolate(
        self, content: str, start: int, end: int,
        blocks: List[Tuple[str, Span]],
    ) -> List[Span]:
        relevant: List[Tuple[str, Span]] = [
            (k, (s, e)) for k, (s, e) in blocks
            if s >= start and e <= end
        ]
        if not relevant:
            return self.__text.split_block(
                content, start, end,
            )
        return self.__dispatch(
            content, start, end, relevant,
        )

    def __dispatch(
        self, content: str, start: int, end: int,
        relevant: List[Tuple[str, Span]],
    ) -> List[Span]:
        spans: List[Span] = []
        pos: int = start
        for _, (bs, be) in relevant:
            if bs < pos:
                continue
            if bs > pos:
                spans.extend(self.__text.split_block(
                    content, pos, bs,
                ))
            if be - bs <= self.__max:
                spans.append((bs, be))
            else:
                spans.extend(self.__text.split_lines(
                    content, bs, be,
                ))
            pos = be
        if pos < end:
            spans.extend(self.__text.split_block(
                content, pos, end,
            ))
        return spans
