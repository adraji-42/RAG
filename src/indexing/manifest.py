import os
import hashlib
from typing import Dict, List, Optional, Set

from .types import ManifestRecord


class ManifestBuilder:
    def __init__(
        self, old_manifest: Optional[Dict[str, ManifestRecord]] = None
    ) -> None:
        self.__old_manifest: Dict[str, ManifestRecord] = (
            dict(old_manifest) if old_manifest else {}
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

    def __compute_hash(
        self, file_path: str, content: Optional[str] = None
    ) -> str:
        if content is not None:
            return hashlib.sha256(content.encode("utf-8")).hexdigest()
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return ""

    def check_status(
        self,
        file_path: str,
        content: Optional[str] = None,
    ) -> str:
        fp: str = os.path.normpath(file_path)
        rec: Optional[ManifestRecord] = self.__old_manifest.get(fp)
        if rec is None:
            return "added"
        mtime: float = self.__get_mtime(fp)
        if mtime == rec["mtime"]:
            self.__manifest[fp] = rec
            return "unchanged"
        digest: str = self.__compute_hash(fp, content)
        if digest and digest == rec["hash"]:
            self.__manifest[fp] = {
                "hash": digest,
                "mtime": mtime,
            }
            return "unchanged"
        return "modified"

    def count_deleted(self, current_files: List[str]) -> int:
        cur: Set[str] = {os.path.normpath(f) for f in current_files}
        return sum(1 for p in self.__old_manifest if p not in cur)

    def record(
        self, file_path: str, content: str
    ) -> None:
        fp: str = os.path.normpath(file_path)
        self.__manifest[fp] = {
            "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "mtime": self.__get_mtime(fp),
        }
