import os
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from pydantic import ValidationError

from .evaluation import Evaluator
from .indexing import Indexer
from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
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
        processed_dir: str = "data/processed"
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
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset: RagDataset = RagDataset.model_validate_json(
                    f.read()
                )
        except (ValidationError, OSError):
            print(f"Failed to read dataset file: {dataset_path}")
            return

        retriever: BM25Retriever = BM25Retriever(processed_dir)
        search_results: List[MinimalSearchResults] = []

        for item in tqdm(
            dataset.rag_questions, desc="Searching", unit="query"
        ):
            sources: List[MinimalSource] = retriever.retrieve(
                item.question, k=k
            )
            search_results.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
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
            f.write(student_results.model_dump_json(indent=2))

        print(f"Saved student_search_results to {out_path}")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        if not os.path.isfile(student_search_results_path):
            print(f"File not found: {student_search_results_path}")
            return
        if not os.path.isfile(dataset_path):
            print(f"File not found: {dataset_path}")
            return

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_results: StudentSearchResults = (
                    StudentSearchResults.model_validate_json(f.read())
                )
        except (ValidationError, OSError):
            print(
                f"Invalid student search results file: "
                f"{student_search_results_path}"
            )
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset: RagDataset = RagDataset.model_validate_json(
                    f.read()
                )
        except (ValidationError, OSError):
            print(f"Invalid dataset file: {dataset_path}")
            return

        evaluator: Evaluator = Evaluator()
        scores: Dict[int, float] = evaluator.evaluate_recall_at_k(
            student_results, dataset, k_values=[1, 3, 5, 10]
        )
        count: int = sum(
            1
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        )

        print("Student data is valid: True")
        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {count}")
        for k in [1, 3, 5, 10]:
            score: float = scores.get(k, 0.0)
            print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")
