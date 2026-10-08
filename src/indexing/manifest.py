import os
import pickle
import hashlib
from typing import Dict, TypedDict


class ManifestRecord(TypedDict):
    hash: str
    mtime: float
    chunk_count: int


class ManifestBuilder:

    def __init__(self) -> None:
        self.__manifest: Dict[str, ManifestRecord] = {}

    @property
    def manifest(self) -> Dict[str, ManifestRecord]:
        return self.__manifest

    def record(
        self, file_path: str, content: str, chunk_count: int
    ) -> None:
        norm_path: str = os.path.normpath(file_path)
        try:
            mtime: float = os.path.getmtime(file_path)
        except OSError:
            mtime = 0.0
        file_hash: str = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        self.__manifest[norm_path] = {
            "hash": file_hash,
            "mtime": mtime,
            "chunk_count": chunk_count,
        }

    def save(self, file_path: str) -> None:
        parent: str = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(
                self.__manifest, f, protocol=pickle.HIGHEST_PROTOCOL,
            )
