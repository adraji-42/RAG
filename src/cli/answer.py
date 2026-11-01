import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .io import CliIO
from ..models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .base import BaseCommand
from ..retrieval import BaseRetriever
from ..generation import AnswerGenerator


class AnswerCommand(BaseCommand):
    def cmd(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        try:
            retriever: BaseRetriever = self.get_retriever(mode, processed_dir)
            sources: List[MinimalSource] = retriever.retrieve(query, k=k)
            for s in sources:
                p1, p2 = s.first_character_index, s.last_character_index
                print(f"{s.file_path} [{p1}:{p2}]")
            gen: AnswerGenerator = AnswerGenerator(processed_dir)
            ans: str = gen.generate(query, sources)
            print(f"\nAnswer:\n{ans}")
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)


class AnswerDatasetCommand(BaseCommand):
    def __generate_answers(
        self, results: StudentSearchResults
    ) -> List[MinimalAnswer]:
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
        try:
            student_results: Optional[StudentSearchResults] = (
                CliIO.load_student_results(student_search_results_path)
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
                save_directory,
                Path(student_search_results_path).name,
                "student_search_results_and_answer",
            )
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
