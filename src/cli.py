import json
import os
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from .indexing import Indexer
from .models import MinimalSearchResults, MinimalSource, StudentSearchResults
from .retrieval import BM25Retriever


class Cli:

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        indexer: Indexer = Indexer(
            max_chunk_size=max_chunk_size,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
        indexer.run()

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
    ) -> None:
        retriever: BM25Retriever = BM25Retriever(processed_dir)
        sources: List[MinimalSource] = retriever.retrieve(query, k)
        for s in sources:
            span: str = (
                f"[{s.first_character_index}:{s.last_character_index}]"
            )
            print(f"{s.file_path} {span}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
    ) -> None:
        if not os.path.isfile(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Failed to read dataset file: {dataset_path}")
            return

        if not isinstance(data, dict):
            print(f"Invalid dataset format in: {dataset_path}")
            return

        raw_questions: Any = data.get("rag_questions", [])
        if not isinstance(raw_questions, list):
            print(f"Invalid rag_questions format in: {dataset_path}")
            return

        retriever: BM25Retriever = BM25Retriever(processed_dir)
        search_results: List[MinimalSearchResults] = []

        for item in tqdm(raw_questions, desc="Searching", unit="query"):
            if not isinstance(item, dict):
                continue
            qid: str = str(item.get("question_id", ""))
            qtext: str = str(item.get("question", ""))
            sources: List[MinimalSource] = retriever.retrieve(qtext, k=k)
            search_results.append(
                MinimalSearchResults(
                    question_id=qid,
                    question=qtext,
                    retrieved_sources=sources,
                )
            )

        student_results: StudentSearchResults = StudentSearchResults(
            search_results=search_results,
            k=k,
        )

        os.makedirs(save_directory, exist_ok=True)
        out_path: str = os.path.join(
            save_directory, Path(dataset_path).name
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(student_results.model_dump(), f, indent=2)

        print(f"Saved student_search_results to {out_path}")
