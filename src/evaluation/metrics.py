from typing import Dict, List

from ..models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def compute_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    inter: int = max(0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    union: int = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def is_match(
    retrieved: MinimalSource,
    ground_truth: MinimalSource,
    min_iou: float = 0.05,
) -> bool:
    if retrieved.file_path != ground_truth.file_path:
        return False
    return (
        compute_iou(
            retrieved.first_character_index,
            retrieved.last_character_index,
            ground_truth.first_character_index,
            ground_truth.last_character_index,
        )
        >= min_iou
    )


class Evaluator:

    def __init__(self, min_iou: float = 0.05) -> None:
        self.__min_iou: float = min_iou

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

        results: Dict[int, float] = {}
        for k in k_values:
            total_recall: float = 0.0
            for qid, gt_q in gt_map.items():
                retrieved_top_k: List[MinimalSource] = (
                    student_map.get(qid, [])[:k]
                )
                found_count: int = 0
                for gt_src in gt_q.sources:
                    if any(
                        is_match(r_src, gt_src, self.__min_iou)
                        for r_src in retrieved_top_k
                    ):
                        found_count += 1
                q_recall: float = (
                    found_count / len(gt_q.sources)
                    if gt_q.sources
                    else 0.0
                )
                total_recall += q_recall
            results[k] = total_recall / len(gt_map)

        return results
