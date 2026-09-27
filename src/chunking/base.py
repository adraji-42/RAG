"""Abstract base class and shared data model for the chunking pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single text chunk with character-level span in the source file."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str

    @classmethod
    def spanning(cls, file_path: str, start: int, text: str) -> Chunk:
        """Build the chunk covering *text* starting at offset *start*."""
        return cls(file_path, start, start + len(text), text)


class BaseChunker(ABC):
    """Strategy interface for all file-type-specific chunking strategies."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialise the chunker with a configurable maximum chunk size.

        Raises:
            ValueError: if `max_chunk_size` is not a positive integer.
        """
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be a positive integer")
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, file_path: Path, content: str) -> Iterator[Chunk]:
        """Yield :class:`Chunk` objects for every segment of *content*."""

    def _read_file(self, file_path: Path) -> str | None:
        """Return file content as a string, or ``None`` on any read error."""
        for encoding in ("utf-8", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except OSError:
                return None
        return None

    def _split_by_size(
        self,
        text: str,
        offset: int,
        file_path: str,
    ) -> Iterator[Chunk]:
        """Yield line-aligned chunks that never exceed ``max_chunk_size``.

        This is the last-resort strategy, used only when a structural
        unit (a markdown block, a paragraph, a function) still
        exceeds the configured size on its own.
        """
        buf_start = offset
        buf: list[str] = []
        buf_len = 0

        for line in text.splitlines(keepends=True):
            line_len = len(line)
            if buf_len + line_len > self.max_chunk_size and buf:
                yield from self._flush_buffer(buf, buf_start, file_path)
                buf_start += buf_len
                buf, buf_len = [], 0
            if line_len > self.max_chunk_size:
                yield from self._split_long_line(line, buf_start, file_path)
                buf_start += line_len
                continue
            buf.append(line)
            buf_len += line_len

        if buf:
            yield from self._flush_buffer(buf, buf_start, file_path)

    def _flush_buffer(
        self,
        buf: list[str],
        buf_start: int,
        file_path: str,
    ) -> Iterator[Chunk]:
        """Emit a single :class:`Chunk` from accumulated buffer lines."""
        yield Chunk.spanning(file_path, buf_start, "".join(buf))

    def _split_long_line(
        self,
        line: str,
        offset: int,
        file_path: str,
    ) -> Iterator[Chunk]:
        """Yield fixed-size slices for a line exceeding ``max_chunk_size``."""
        for i in range(0, len(line), self.max_chunk_size):
            part = line[i: i + self.max_chunk_size]
            yield Chunk.spanning(file_path, offset + i, part)

    def chunk_file(self, file_path: str) -> Iterator[Chunk]:
        """Read *file_path* from disk and delegate to :meth:`chunk`."""
        path = Path(file_path)
        content = self._read_file(path)
        if content is None:
            return
        yield from self.chunk(path, content)
