import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .storage import IndexStorage
from .manifest import ManifestBuilder
from .pipeline import ChunkingPipeline
from .types import ChunkRecord, ManifestRecord
from .embeddings import EmbeddingModel, VectorIndex


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
    ) -> Tuple[
        Dict[str, ManifestRecord],
        Dict[str, List[ChunkRecord]],
        Dict[str, List[np.ndarray]],
    ]:
        manifest: Dict[str, ManifestRecord] = self.__storage.load_manifest()
        chunks: List[ChunkRecord] = self.__storage.load_chunks()
        embs: np.ndarray = self.__storage.load_embeddings().embeddings
        grp_c: Dict[str, List[ChunkRecord]] = {}
        grp_e: Dict[str, List[np.ndarray]] = {}
        for c, e in zip(chunks, embs, strict=True):
            fp: str = os.path.normpath(c["source"].file_path)
            grp_c.setdefault(fp, []).append(c)
            grp_e.setdefault(fp, []).append(e)
        return manifest, grp_c, grp_e

    def __save_state(
        self,
        chunks: List[ChunkRecord],
        manifest: Dict[str, ManifestRecord],
        vector_index: VectorIndex,
        changed: bool,
    ) -> None:
        if changed:
            self.__storage.save_artifacts(chunks, manifest, vector_index)
        else:
            self.__storage.save_manifest(manifest)

    def __print_summary(
        self, counts: Dict[str, int], deleted: int, total: int
    ) -> None:
        print(
            f"Incremental stats: +{counts['added']} added, "
            f"~{counts['modified']} modified, -{deleted} deleted, "
            f"={counts['unchanged']} unchanged"
        )
        print(
            f"Ingestion complete! Indexed {total} "
            f"chunks under {self.__storage.processed_dir}"
        )

    def __build_embeddings(
        self, emb_list: List[np.ndarray]
    ) -> VectorIndex:
        if emb_list:
            return VectorIndex(np.vstack(emb_list).astype(np.float32))
        return VectorIndex(np.empty((0, 384), dtype=np.float32))

    def __process_file(
        self,
        fp: str,
        old_grp: Dict[str, List[ChunkRecord]],
        old_embs: Dict[str, List[np.ndarray]],
        bld: ManifestBuilder,
        counts: Dict[str, int],
        model: Optional[EmbeddingModel],
    ) -> Tuple[
        List[ChunkRecord], Optional[np.ndarray], Optional[EmbeddingModel]
    ]:
        c_chk: List[ChunkRecord] = old_grp.get(fp, [])
        c_emb: List[np.ndarray] = old_embs.get(fp, [])
        status: str = bld.check_status(fp)
        counts[status] += 1
        if status == "unchanged":
            emb: Optional[np.ndarray] = (
                np.array(c_emb, dtype=np.float32) if c_emb else None
            )
            return c_chk, emb, model
        nc: List[ChunkRecord] = self.__pipeline.chunk_file(fp, bld)
        if not nc:
            return nc, None, model
        if model is None:
            model = EmbeddingModel()
        return nc, model.encode([c["text"] for c in nc]), model

    def run(self) -> None:
        old_man, old_grp, old_embs = self.__load_state()
        bld: ManifestBuilder = ManifestBuilder(old_man)
        files: List[str] = self.__pipeline.discover_files()
        counts: Dict[str, int] = {"added": 0, "modified": 0, "unchanged": 0}
        chunks: List[ChunkRecord] = []
        emb_list: List[np.ndarray] = []
        model: Optional[EmbeddingModel] = None
        for fp in tqdm(files, desc="Indexing", unit="file", leave=False):
            chks, emb, model = self.__process_file(
                fp, old_grp, old_embs, bld, counts, model
            )
            chunks.extend(chks)
            if emb is not None and len(emb) > 0:
                emb_list.append(emb)
        deleted: int = bld.count_deleted(files)
        changed: bool = bool(counts["added"] or counts["modified"] or deleted)
        self.__pipeline.audit(chunks)
        vector_index: VectorIndex = self.__build_embeddings(emb_list)
        self.__save_state(chunks, bld.manifest, vector_index, changed)
        self.__print_summary(counts, deleted, len(chunks))
