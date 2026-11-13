"""CLI commands for single query answering and batch dataset answering."""

import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from pydantic import ValidationError

from .io import CliIO
from .base import BaseCommand
from ..retrieval import BaseRetriever
from ..generation import AnswerGenerator
from .models import AnswerDatasetOptions, AnswerOptions
from ..models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class AnswerCommand(BaseCommand):
    """Execute CLI query answering using retrieval and generation."""

    def cmd(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        """Retrieve sources and generate an answer for a single query.

        Args:
            query: Query question text string.
            k: Maximum number of source spans to retrieve.
            processed_dir: Path to directory containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
        """
        try:
            options: AnswerOptions = AnswerOptions(
                query=query,
                k=k,
                processed_dir=processed_dir,
                mode=mode,
            )
            retriever: BaseRetriever = self.get_retriever(
                options.mode, options.processed_dir
            )
            sources: List[MinimalSource] = retriever.retrieve(
                options.query, k=options.k
            )
            gen: AnswerGenerator = AnswerGenerator(
                processed_dir=options.processed_dir
            )
            ans: str = gen.generate(options.query, sources)
            print(f"\nAnswer:\n{ans}")
        except ValidationError as e:
            print(self.format_validation_error(e), file=sys.stderr)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)


class AnswerDatasetCommand(BaseCommand):
    """Execute batch answer generation over precomputed search results."""

    def __generate_answers(
        self, results: StudentSearchResults
    ) -> List[MinimalAnswer]:
        """Generate answers sequentially for a batch of search results.

        Args:
            results: StudentSearchResults containing retrieved sources.

        Returns:
            List of MinimalAnswer objects with generated answers.
        """
        generator: AnswerGenerator = AnswerGenerator()
        answers: List[MinimalAnswer] = []
        for item in tqdm(
            results.search_results,
            desc="Generating answers", unit="q", leave=False
        ):
            ans: str = generator.generate(
                item.question, item.retrieved_sources
            )
            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=ans,
                )
            )
        return answers

    def cmd(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """Load batch search results, generate answers, and save to output.

        Args:
            student_search_results_path: Path to search results JSON file.
            save_directory: Directory path to save output answer file.
        """
        try:
            options: AnswerDatasetOptions = AnswerDatasetOptions(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
            )
            student_results: Optional[StudentSearchResults] = (
                CliIO.load_student_results(
                    options.student_search_results_path
                )
            )
            if student_results is None:
                return
            answers: List[MinimalAnswer] = self.__generate_answers(
                student_results
            )
            CliIO.save_result_file(
                StudentSearchResultsAndAnswer(
                    search_results=answers,
                    k=student_results.k,
                ),
                options.save_directory,
                Path(options.student_search_results_path).name,
                "student_search_results_and_answer",
            )
        except ValidationError as e:
            print(self.format_validation_error(e), file=sys.stderr)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
