from typing import List, Tuple
from abc import ABC, abstractmethod

from ..models import MinimalSource


class BaseChunker(ABC):

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self._max_chunk_size: int = max_chunk_size

    @abstractmethod
    def chunk(
        self,
        file_path: str,
        content: str,
    ) -> List[MinimalSource]:
        ...

    def _build_line_offsets(
        self,
        content: str,
    ) -> List[int]:
        offsets: List[int] = [0]
        for i, ch in enumerate(content):
            if ch == "\n":
                offsets.append(i + 1)
        return offsets

    def _find_newline_split(
        self,
        content: str,
        start: int,
        end: int,
    ) -> int:
        limit: int = start + self._max_chunk_size
        candidate: int = content.rfind("\n", start, limit)
        if candidate > start:
            return candidate + 1
        return limit

    def _fallback_split(
        self,
        file_path: str,
        content: str,
        start: int,
        end: int,
    ) -> List[MinimalSource]:
        chunks: List[MinimalSource] = []
        pos: int = start
        while pos < end:
            if end - pos <= self._max_chunk_size:
                chunks.append(self._make_source(
                    file_path, pos, end,
                ))
                break
            split: int = self._find_newline_split(
                content, pos, end,
            )
            split = min(split, end)
            chunks.append(self._make_source(
                file_path, pos, split,
            ))
            pos = split
        return chunks

    def _emit_or_split(
        self,
        file_path: str,
        content: str,
        start: int,
        end: int,
    ) -> List[MinimalSource]:
        size: int = end - start
        if size <= 0:
            return []
        if size <= self._max_chunk_size:
            return [self._make_source(
                file_path, start, end,
            )]
        return self._fallback_split(
            file_path, content, start, end,
        )

    def _make_source(
        self,
        file_path: str,
        start: int,
        end: int,
    ) -> MinimalSource:
        return MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
        )

    def _pack_spans(
        self,
        content: str,
        spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not spans:
            return []
        packed: List[Tuple[int, int]] = []
        cur_start, cur_end = spans[0]
        for start, end in spans[1:]:
            merged: int = end - cur_start
            gap_empty: bool = (
                content[cur_end:start].strip() == ""
            )
            if gap_empty and merged <= self._max_chunk_size:
                cur_end = end
            else:
                packed.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        packed.append((cur_start, cur_end))
        return packed

    def _is_whitespace_only(
        self,
        content: str,
        start: int,
        end: int,
    ) -> bool:
        return content[start:end].strip() == ""

    def _filter_empty_spans(
        self,
        content: str,
        spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        return [
            (s, e) for s, e in spans
            if s < e and not self._is_whitespace_only(
                content, s, e,
            )
        ]

    def _spans_to_chunks(
        self,
        file_path: str,
        content: str,
        spans: List[Tuple[int, int]],
    ) -> List[MinimalSource]:
        clean: List[Tuple[int, int]] = (
            self._filter_empty_spans(content, spans)
        )
        packed: List[Tuple[int, int]] = self._pack_spans(
            content, clean,
        )
        chunks: List[MinimalSource] = []
        for start, end in packed:
            chunks.extend(
                self._emit_or_split(
                    file_path, content, start, end,
                )
            )
        return chunks
