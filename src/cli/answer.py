import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .io import CliIO
from .base import BaseCommand
from ..retrieval import BM25Retriever
from ..generation import AnswerGenerator
from ..models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class AnswerCommand(BaseCommand):

    def cmd(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
    ) -> None:
        if not query.strip() or k <= 0:
            print("Invalid query or k value.", file=sys.stderr)
            return
        retriever: BM25Retriever = BM25Retriever(processed_dir)
        sources: List[MinimalSource] = retriever.retrieve(query, k=k)
        for s in sources:
            span: str = (
                f"[{s.first_character_index}:{s.last_character_index}]"
            )
            print(f"{s.file_path} {span}")
        generator: AnswerGenerator = AnswerGenerator()
        ans: str = generator.generate(query, sources)
        print(f"\nAnswer:\n{ans}")


class AnswerDatasetCommand(BaseCommand):

    def _generate_answers(
        self, results: StudentSearchResults
    ) -> List[MinimalAnswer]:
        generator: AnswerGenerator = AnswerGenerator()
        answers: List[MinimalAnswer] = []
        for item in tqdm(
            results.search_results, desc="Generating answers", unit="q"
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
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        student_results: Optional[StudentSearchResults] = (
            CliIO.load_student_results(student_search_results_path)
        )
        if student_results is None:
            return
        answers: List[MinimalAnswer] = self._generate_answers(
            student_results
        )
        result: StudentSearchResultsAndAnswer = (
            StudentSearchResultsAndAnswer(
                search_results=answers,
                k=student_results.k,
            )
        )
        CliIO.save_result_file(
            result,
            save_directory,
            Path(student_search_results_path).name,
            "student_search_results_and_answer",
        )
