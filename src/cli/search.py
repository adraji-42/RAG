from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from .io import CliIO
from .base import BaseCommand
from ..retrieval import BaseRetriever, BM25Retriever, HybridRetriever


class SearchCommand(BaseCommand):
    def cmd(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        retriever: BaseRetriever = (
            HybridRetriever(processed_dir)
            if mode == "hybrid"
            else BM25Retriever(processed_dir)
        )
        sources: List[MinimalSource] = retriever.retrieve(query, k)
        for s in sources:
            span: str = (
                f"[{s.first_character_index}:{s.last_character_index}]"
            )
            print(f"{s.file_path} {span}")


class SearchDatasetCommand(BaseCommand):
    def __search_queries(
        self, dataset: RagDataset, k: int, processed_dir: str, mode: str
    ) -> List[MinimalSearchResults]:
        retriever: BaseRetriever = (
            HybridRetriever(processed_dir)
            if mode == "hybrid"
            else BM25Retriever(processed_dir)
        )
        results: List[MinimalSearchResults] = []
        for item in tqdm(
            dataset.rag_questions, desc="Searching", unit="query"
        ):
            sources: List[MinimalSource] = retriever.retrieve(
                item.question, k=k
            )
            results.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=sources,
                )
            )
        return results

    def cmd(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        dataset: Optional[RagDataset] = CliIO.load_dataset(dataset_path)
        if dataset is None:
            return
        results: List[MinimalSearchResults] = self.__search_queries(
            dataset, k, processed_dir, mode
        )
        student_results: StudentSearchResults = StudentSearchResults(
            search_results=results, k=k
        )
        CliIO.save_result_file(
            student_results,
            save_directory,
            Path(dataset_path).name,
            "student_search_results",
        )
