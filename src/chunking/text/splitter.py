import re
from typing import List

from ..base import Span

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.?!]\s+")


class RecursiveTextSplitter:

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.__max: int = max_chunk_size

    def split_block(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        if end - start <= self.__max:
            return [(start, end)]
        spans: List[Span] = []
        for ps, pe in self.__split_paragraphs(content, start, end):
            if pe - ps <= self.__max:
                spans.append((ps, pe))
            else:
                for ss, se in self.__split_sentences(content, ps, pe):
                    if se - ss <= self.__max:
                        spans.append((ss, se))
                    else:
                        spans.extend(self.split_lines(content, ss, se))
        return spans

    def split_lines(
        self, content: str, start: int, end: int,
    ) -> List[Span]:
        pos: int = start
        spans: List[Span] = []
        for line in content[start:end].split("\n"):
            line_end: int = min(pos + len(line) + 1, end)
            spans.append((pos, line_end))
            pos = line_end
        return spans

    def __split_paragraphs(
        self, text: str, start: int, end: int,
    ) -> List[Span]:
        matches = list(_PARAGRAPH_SPLIT.finditer(text, start, end))
        if not matches:
            return [(start, end)]
        pos: int = start
        spans: List[Span] = []
        for m in matches:
            span_end: int = m.end()
            if span_end > pos:
                spans.append((pos, min(span_end, end)))
                pos = span_end
        if pos < end:
            spans.append((pos, end))
        return spans

    def __split_sentences(
        self, text: str, start: int, end: int,
    ) -> List[Span]:
        spans: List[Span] = []
        pos: int = start
        for m in _SENTENCE_END.finditer(text, start, end):
            if m.end() > pos:
                spans.append((pos, m.end()))
                pos = m.end()
        if pos < end:
            spans.append((pos, end))
        return spans
