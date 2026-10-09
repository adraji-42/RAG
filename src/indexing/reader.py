import os
from pathlib import Path
from typing import List, Set


class CorpusReader:

    def __init__(
        self, raw_dir: str, supported_extensions: List[str]
    ) -> None:
        self.__raw_dir: str = raw_dir
        self.__supported_extensions: Set[str] = set(supported_extensions)

    def discover_files(self) -> List[str]:
        found: List[str] = []
        for root, _, names in os.walk(self.__raw_dir):
            for name in names:
                if Path(name).suffix in self.__supported_extensions:
                    found.append(os.path.normpath(os.path.join(root, name)))
        return sorted(found)

    def read_file(self, file_path: str) -> str:
        try:
            with open(
                file_path, "r", encoding="utf-8", errors="replace", newline=""
            ) as f:
                return f.read()
        except OSError:
            return ""
