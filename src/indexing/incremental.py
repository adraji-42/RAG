import os
import hashlib
from typing import Dict, List, Tuple

from tqdm import tqdm

from .bm25 import BM25Index
from .storage import IndexStorage
from .manifest import ManifestBuilder
from .pipeline import ChunkingPipeline
from .types import ChunkRecord, ManifestRecord


class IncrementalIndexer:
    def __init__(
        self,
        pipeline: ChunkingPipeline,
        storage: IndexStorage,
    ) -> None:
        self.__pipeline: ChunkingPipeline = pipeline
        self.__storage: IndexStorage = storage

    def __load_state(
        self, force: bool
    ) -> Tuple[Dict[str, ManifestRecord], Dict[str, List[ChunkRecord]]]:
        if force:
            return {}, {}
        try:
            manifest: Dict[str, ManifestRecord] = (
                self.__storage.load_manifest()
            )
            chunks: List[ChunkRecord] = self.__storage.load_chunks()
        except (OSError, EOFError):
            return {}, {}
        grouped: Dict[str, List[ChunkRecord]] = {}
        for c in chunks:
            fp: str = os.path.normpath(c["source"].file_path)
            grouped.setdefault(fp, []).append(c)
        return manifest, grouped

    def __get_mtime(self, fp: str) -> float:
        try:
            return os.path.getmtime(fp)
        except OSError:
            return 0.0

    def __process_file(
        self,
        fp: str,
        old_manifest: Dict[str, ManifestRecord],
        old_grouped: Dict[str, List[ChunkRecord]],
        new_manifest: ManifestBuilder,
    ) -> Tuple[str, List[ChunkRecord]]:
        norm_fp: str = os.path.normpath(fp)
        if norm_fp not in old_manifest:
            return "added", self.__pipeline.chunk_file(fp, new_manifest)
        rec: ManifestRecord = old_manifest[norm_fp]
        cached: List[ChunkRecord] = old_grouped.get(norm_fp, [])
        mtime: float = self.__get_mtime(fp)
        if mtime == rec["mtime"] and len(cached) == rec["chunk_count"]:
            new_manifest.manifest[norm_fp] = rec
            return "unchanged", cached
        content: str = self.__pipeline.read_file(fp)
        digest: str = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest == rec["hash"] and len(cached) == rec["chunk_count"]:
            new_manifest.manifest[norm_fp] = {
                "hash": digest,
                "mtime": mtime,
                "chunk_count": rec["chunk_count"],
            }
            return "unchanged", cached
        return "modified", self.__pipeline.chunk_file(fp, new_manifest)

    def __save_state(
        self,
        chunks: List[ChunkRecord],
        manifest: Dict[str, ManifestRecord],
        changed: bool,
    ) -> None:
        if not changed and os.path.isfile(self.__storage.bm25_path):
            self.__storage.save_manifest(manifest)
            return
        self.__storage.save_chunks(chunks)
        self.__storage.save_manifest(manifest)
        bm25: BM25Index = BM25Index([c["text"] for c in chunks])
        self.__storage.save_bm25(bm25)

    def run(self, force: bool = False) -> List[ChunkRecord]:
        old_man, old_grp = self.__load_state(force)
        new_man: ManifestBuilder = ManifestBuilder()
        files: List[str] = self.__pipeline.discover_files()
        counts: Dict[str, int] = {
            "added": 0, "modified": 0, "unchanged": 0
        }
        chunks: List[ChunkRecord] = []
        for fp in tqdm(files, desc="Indexing", unit="file"):
            status, file_chunks = self.__process_file(
                fp, old_man, old_grp, new_man
            )
            counts[status] += 1
            chunks.extend(file_chunks)
        cur_set = {os.path.normpath(f) for f in files}
        deleted: int = sum(1 for k in old_man if k not in cur_set)
        changed: bool = bool(
            force or counts["added"] or counts["modified"] or deleted
        )
        self.__pipeline.audit(chunks)
        self.__save_state(chunks, new_man.manifest, changed)
        print(
            f"Incremental stats: +{counts['added']} added, "
            f"~{counts['modified']} modified, -{deleted} deleted, "
            f"={counts['unchanged']} unchanged"
        )
        print(
            f"Ingestion complete! Indexed {len(chunks)} "
            f"chunks under {self.__storage.processed_dir}"
        )
        return chunks
