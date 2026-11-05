"""Corpus file reader and extension-based file discovery utility."""

import os
from pathlib import Path
from typing import List, Set


class CorpusReader:
    """Discover and read supported document files from corpus directory."""

    def __init__(
        self, raw_dir: str, supported_extensions: List[str]
    ) -> None:
        """Initialize the corpus reader with directory path and extensions.

        Args:
            raw_dir: Root directory path containing document files.
            supported_extensions: List of supported file extensions.

        Raises:
            FileNotFoundError: If raw_dir does not exist as a directory.
        """
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        self.__raw_dir: str = raw_dir
        self.__supported_extensions: Set[str] = set(supported_extensions)

    def discover_files(self) -> List[str]:
        """Traverse directory and return sorted list of file paths.

        Returns:
            Sorted list of normalized file paths for supported documents.
        """
        found: List[str] = []
        for root, _, names in os.walk(self.__raw_dir):
            for name in names:
                if Path(name).suffix in self.__supported_extensions:
                    found.append(os.path.normpath(os.path.join(root, name)))
        return sorted(found)

    def read_file(self, file_path: str) -> str:
        """Read and decode file contents using UTF-8 encoding.

        Args:
            file_path: Path to target file on disk.

        Returns:
            UTF-8 decoded file content string.

        Raises:
            OSError: If reading the file fails.
        """
        try:
            with open(
                file_path, "r", encoding="utf-8", errors="replace", newline=""
            ) as f:
                return f.read()
        except OSError:
            raise OSError(f"Failed to read file: {file_path}")
