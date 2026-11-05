"""CLI command evaluating student search results against ground truth."""

import sys
from typing import Dict, Optional

from .io import CliIO
from .base import BaseCommand
from ..evaluation import Evaluator
from ..models import AnsweredQuestion, RagDataset, StudentSearchResults


class EvaluateCommand(BaseCommand):
    """Evaluate student retrieval results against ground-truth dataset."""

    def __print_eval_report(
        self, scores: Dict[int, float], count: int
    ) -> None:
        """Print formatted recall scores table for evaluated questions.

        Args:
            scores: Dictionary mapping k cutoff integers to recall scores.
            count: Total number of evaluated answered questions.
        """
        print("Student data is valid: True")
        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {count}")
        for k in [1, 3, 5, 10]:
            score: float = scores.get(k, 0.0)
            print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")

    def cmd(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Evaluate student results against ground truth and print report.

        Args:
            student_search_results_path: Path to student search results JSON.
            dataset_path: Path to ground-truth evaluation dataset JSON file.
        """
        try:
            student_results: Optional[StudentSearchResults] = (
                CliIO.load_student_results(student_search_results_path)
            )
            if student_results is None:
                return
            dataset: Optional[RagDataset] = CliIO.load_eval_dataset(
                dataset_path
            )
            if dataset is None:
                return
            evaluator: Evaluator = Evaluator()
            scores: Dict[int, float] = evaluator.evaluate(
                student_results, dataset, k_values=[1, 3, 5, 10]
            )
            count: int = sum(
                1
                for q in dataset.rag_questions
                if isinstance(q, AnsweredQuestion) and q.sources
            )
            self.__print_eval_report(scores, count)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
