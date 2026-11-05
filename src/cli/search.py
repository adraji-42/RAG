"""CLI search commands for single queries and batch evaluation datasets."""

import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .io import CliIO
from ..models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from .base import BaseCommand
from ..retrieval import BaseRetriever


class SearchCommand(BaseCommand):
    """Execute CLI query search across indexed documents."""

    def cmd(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        """Retrieve and print matching sources for a single query.

        Args:
            query: Query question text string.
            k: Maximum number of source spans to retrieve.
            processed_dir: Path to directory containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
        """
        try:
            retriever: BaseRetriever = self.get_retriever(mode, processed_dir)
            sources: List[MinimalSource] = retriever.retrieve(query, k)
            for s in sources:
                p1, p2 = s.first_character_index, s.last_character_index
                print(f"{s.file_path} [{p1}:{p2}]")
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)


class SearchDatasetCommand(BaseCommand):
    """Execute batch dataset query retrieval and save results."""

    def __search_queries(
        self, dataset: RagDataset, k: int, processed_dir: str, mode: str
    ) -> List[MinimalSearchResults]:
        """Retrieve sources for all questions in the dataset.

        Args:
            dataset: RagDataset containing questions to evaluate.
            k: Maximum number of source spans to retrieve per question.
            processed_dir: Path to directory containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').

        Returns:
            List of MinimalSearchResults with retrieved sources.
        """
        retriever: BaseRetriever = self.get_retriever(mode, processed_dir)
        results: List[MinimalSearchResults] = []
        for item in tqdm(
            dataset.rag_questions, desc="Searching", unit="query", leave=False
        ):
            try:
                sources: List[MinimalSource] = retriever.retrieve(
                    item.question, k=k
                )
            except ValueError as e:
                print(
                    f"Warning: skipping invalid query "
                    f"'{item.question_id}': {e}", file=sys.stderr
                )
                sources = []
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
        """Load dataset, execute batch query search, and save results.

        Args:
            dataset_path: Path to dataset JSON file.
            k: Maximum number of source spans to retrieve per question.
            save_directory: Directory where output JSON will be saved.
            processed_dir: Path to directory containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
        """
        try:
            if k <= 0:
                raise ValueError("k must be greater than 0")
            dataset: Optional[RagDataset] = CliIO.load_dataset(dataset_path)
            if dataset is None:
                return
            results: List[MinimalSearchResults] = (
                self.__search_queries(dataset, k, processed_dir, mode)
            )
            CliIO.save_result_file(
                StudentSearchResults(search_results=results, k=k),
                save_directory, Path(dataset_path).name,
                "student_search_results",
            )
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
