import os
import hashlib
from typing import Dict, Optional

from .types import ManifestRecord


class ManifestBuilder:
    def __init__(
        self, initial: Optional[Dict[str, ManifestRecord]] = None
    ) -> None:
        self.__initial: Dict[str, ManifestRecord] = (
            dict(initial) if initial else {}
        )
        self.__manifest: Dict[str, ManifestRecord] = {}

    @property
    def manifest(self) -> Dict[str, ManifestRecord]:
        return self.__manifest

    def __get_mtime(self, file_path: str) -> float:
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0.0

    def __compute_hash(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return ""

    def check_status(self, file_path: str, cached_count: int) -> str:
        norm_path: str = os.path.normpath(file_path)
        rec: Optional[ManifestRecord] = self.__initial.get(norm_path)
        if rec is None:
            return "added"
        if cached_count != rec["chunk_count"]:
            return "modified"
        mtime: float = self.__get_mtime(file_path)
        if mtime == rec["mtime"]:
            self.__manifest[norm_path] = rec
            return "unchanged"
        digest: str = self.__compute_hash(file_path)
        if digest and digest == rec["hash"]:
            self.__manifest[norm_path] = {
                "hash": digest,
                "mtime": mtime,
                "chunk_count": rec["chunk_count"],
            }
            return "unchanged"
        return "modified"

    def record(
        self, file_path: str, content: str, chunk_count: int
    ) -> None:
        norm_path: str = os.path.normpath(file_path)
        self.__manifest[norm_path] = {
            "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "mtime": self.__get_mtime(file_path),
            "chunk_count": chunk_count,
        }
