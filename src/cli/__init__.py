from .indexing import IndexCommand
from .evaluation import EvaluateCommand
from .search import SearchCommand, SearchDatasetCommand


class Cli:
    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        force: bool = False,
    ) -> None:
        IndexCommand().cmd(max_chunk_size, raw_dir, processed_dir, force)

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        SearchCommand().cmd(query, k, processed_dir, mode)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        SearchDatasetCommand().cmd(
            dataset_path, k, save_directory, processed_dir, mode
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        EvaluateCommand().cmd(student_search_results_path, dataset_path)

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        from .answer import AnswerCommand

        AnswerCommand().cmd(query, k, processed_dir, mode)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        mode: str = "bm25",
    ) -> None:
        from .answer import AnswerDatasetCommand

        AnswerDatasetCommand().cmd(
            student_search_results_path, save_directory, mode
        )


__all__ = ["Cli"]
