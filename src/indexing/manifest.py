"""Corpus manifest builder tracking file hashes and modification timestamps."""

import os
import hashlib
from typing import Dict, List, Optional, Set

from .types import ManifestRecord


class ManifestBuilder:
    """Build and track file hashes and modification timestamps."""

    def __init__(
        self, old_manifest: Optional[Dict[str, ManifestRecord]] = None
    ) -> None:
        """Initialize manifest builder with an optional previous manifest.

        Args:
            old_manifest: Optional previous manifest dictionary.
        """
        self.__old_manifest: Dict[str, ManifestRecord] = (
            dict(old_manifest) if old_manifest else {}
        )
        self.__manifest: Dict[str, ManifestRecord] = {}

    @property
    def manifest(self) -> Dict[str, ManifestRecord]:
        """Return the accumulated manifest dictionary.

        Returns:
            Dictionary mapping normalized file paths to manifest records.
        """
        return self.__manifest

    def __get_mtime(self, file_path: str) -> float:
        """Return the modification time of a file or 0.0 on error.

        Args:
            file_path: Path to target file on disk.

        Returns:
            Modification timestamp float, or 0.0 on error.
        """
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0.0

    def __compute_hash(
        self, file_path: str, content: Optional[str] = None
    ) -> str:
        """Compute SHA-256 hash of file content or disk file.

        Args:
            file_path: Path to target file on disk.
            content: Optional in-memory string content to hash.

        Returns:
            SHA-256 hexadecimal hash string.
        """
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
        """Check whether a file is added, modified, or unchanged.

        Args:
            file_path: Path to target file.
            content: Optional in-memory string content.

        Returns:
            Status string: 'added', 'modified', or 'unchanged'.
        """
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
        """Count files in old manifest missing from current files.

        Args:
            current_files: List of file paths currently existing on disk.

        Returns:
            Number of files in old manifest missing from current files.
        """
        cur: Set[str] = {os.path.normpath(f) for f in current_files}
        return sum(1 for p in self.__old_manifest if p not in cur)

    def record(
        self, file_path: str, content: str
    ) -> None:
        """Record hash and modification time for a newly chunked file.

        Args:
            file_path: Path to indexed file.
            content: In-memory string content of the file.
        """
        fp: str = os.path.normpath(file_path)
        self.__manifest[fp] = {
            "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "mtime": self.__get_mtime(fp),
        }
