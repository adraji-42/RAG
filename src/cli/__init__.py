from .evaluation import EvaluateCommand
from .indexing import IndexCommand
from .search import SearchCommand, SearchDatasetCommand


class Cli:

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Process and index raw documents for retrieval."""
        IndexCommand().cmd(max_chunk_size, raw_dir, processed_dir)

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
    ) -> None:
        """Search the indexed corpus with a single query."""
        SearchCommand().cmd(query, k, processed_dir)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
    ) -> None:
        """Search questions in a dataset and save retrieved sources."""
        SearchDatasetCommand().cmd(
            dataset_path, k, save_directory, processed_dir
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Evaluate retrieval Recall@k against ground-truth dataset."""
        EvaluateCommand().cmd(student_search_results_path, dataset_path)

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
    ) -> None:
        """Retrieve relevant context and generate an answer using LLM."""
        from .answer import AnswerCommand

        AnswerCommand().cmd(query, k, processed_dir)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for student search results and save output."""
        from .answer import AnswerDatasetCommand

        AnswerDatasetCommand().cmd(student_search_results_path, save_directory)


__all__ = ["Cli"]
