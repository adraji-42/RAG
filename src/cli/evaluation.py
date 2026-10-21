from typing import Dict, Optional

from ..evaluation import Evaluator
from ..models import AnsweredQuestion, RagDataset, StudentSearchResults
from .base import BaseCommand
from .io import CliIO


class EvaluateCommand(BaseCommand):

    def __print_eval_report(
        self, scores: Dict[int, float], count: int
    ) -> None:
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
        scores: Dict[int, float] = evaluator.evaluate_recall_at_k(
            student_results, dataset, k_values=[1, 3, 5, 10]
        )
        count: int = sum(
            1
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        )
        self.__print_eval_report(scores, count)
