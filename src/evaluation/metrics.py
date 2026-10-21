from typing import Dict, List

from ..models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluator:

    def __init__(self, min_iou: float = 0.05) -> None:
        self.__min_iou: float = min_iou

    @staticmethod
    def compute_iou(s1: int, e1: int, s2: int, e2: int) -> float:
        inter: int = max(0, min(e1, e2) - max(s1, s2))
        if inter == 0:
            return 0.0
        union: int = (e1 - s1) + (e2 - s2) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def is_match(
        retrieved: MinimalSource,
        ground_truth: MinimalSource,
        min_iou: float = 0.05,
    ) -> bool:
        if retrieved.file_path != ground_truth.file_path:
            return False
        return (
            Evaluator.compute_iou(
                retrieved.first_character_index,
                retrieved.last_character_index,
                ground_truth.first_character_index,
                ground_truth.last_character_index,
            )
            >= min_iou
        )

    def __recall_for_question(
        self,
        gt_sources: List[MinimalSource],
        retrieved: List[MinimalSource],
    ) -> float:
        if not gt_sources:
            return 0.0
        found: int = sum(
            1
            for gt in gt_sources
            if any(self.is_match(r, gt, self.__min_iou) for r in retrieved)
        )
        return found / len(gt_sources)

    def __mean_recall_at_k(
        self,
        k: int,
        gt_map: Dict[str, AnsweredQuestion],
        student_map: Dict[str, List[MinimalSource]],
    ) -> float:
        total: float = sum(
            self.__recall_for_question(
                gt_q.sources, student_map.get(qid, [])[:k]
            )
            for qid, gt_q in gt_map.items()
        )
        return total / len(gt_map)

    def evaluate_recall_at_k(
        self,
        student_results: StudentSearchResults,
        ground_truth: RagDataset,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[int, float]:
        gt_map: Dict[str, AnsweredQuestion] = {
            q.question_id: q
            for q in ground_truth.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }
        if not gt_map:
            return {k: 0.0 for k in k_values}

        student_map: Dict[str, List[MinimalSource]] = {
            r.question_id: r.retrieved_sources
            for r in student_results.search_results
        }
        return {
            k: self.__mean_recall_at_k(k, gt_map, student_map)
            for k in k_values
        }
