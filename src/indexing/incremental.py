import os
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

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
        self,
    ) -> Tuple[Dict[str, ManifestRecord], Dict[str, List[ChunkRecord]]]:
        manifest: Dict[str, ManifestRecord] = self.__storage.load_manifest()
        chunks: List[ChunkRecord] = self.__storage.load_chunks()
        grouped: Dict[str, List[ChunkRecord]] = {}
        for c in chunks:
            fp: str = os.path.normpath(c["source"].file_path)
            grouped.setdefault(fp, []).append(c)
        return manifest, grouped

    def __save_state(
        self,
        chunks: List[ChunkRecord],
        manifest: Dict[str, ManifestRecord],
        changed: bool,
    ) -> None:
        self.__storage.save_manifest(manifest)
        if changed:
            self.__storage.save_artifacts(chunks, manifest)

    def run(self) -> List[ChunkRecord]:
        old_man, old_grp = self.__load_state()
        manifest_builder: ManifestBuilder = ManifestBuilder(old_man)
        files: List[str] = self.__pipeline.discover_files()
        counts: Dict[str, int] = {
            "added": 0, "modified": 0, "unchanged": 0
        }
        chunks: List[ChunkRecord] = []
        for fp in tqdm(files, desc="Indexing", unit="file"):
            cached: List[ChunkRecord] = old_grp.get(os.path.normpath(fp), [])
            status: str = manifest_builder.check_status(fp, len(cached))
            counts[status] += 1
            if status == "unchanged":
                chunks.extend(cached)
            else:
                chunks.extend(
                    self.__pipeline.chunk_file(fp, manifest_builder)
                )
        cur_files: Set[str] = {os.path.normpath(f) for f in files}
        deleted: int = sum(1 for k in old_man if k not in cur_files)
        changed: bool = bool(
            counts["added"] or counts["modified"] or deleted
        )
        self.__pipeline.audit(chunks)
        self.__save_state(chunks, manifest_builder.manifest, changed)
        print(
            f"Incremental stats: +{counts['added']} added, "
            f"~{counts['modified']} modified, -{deleted} deleted, "
            f"={counts['unchanged']} unchanged"
        )
        return chunks
