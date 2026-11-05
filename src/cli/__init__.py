"""CLI entrypoints and command dispatcher for RAG operations."""

from typing import List

from .serve import ServeCommand
from .indexing import IndexCommand
from .evaluation import EvaluateCommand
from .search import SearchCommand, SearchDatasetCommand


class Cli:
    """Command-line interface root dispatcher for RAG operations."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        force: bool = False,
        embed: bool = False,
    ) -> None:
        """Run full or incremental indexing on corpus documents.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
            raw_dir: Corpus directory path containing raw documents.
            processed_dir: Output directory path for processed index files.
            force: Force full corpus re-indexing from scratch.
            embed: Whether to compute dense vector embeddings.
        """
        IndexCommand().cmd(
            max_chunk_size, raw_dir, processed_dir, force, embed
        )

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        """Execute single query search and print retrieved sources.

        Args:
            query: Search query text string.
            k: Maximum number of retrieved sources to display.
            processed_dir: Output directory path containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
        """
        SearchCommand().cmd(query, k, processed_dir, mode)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        """Run batch evaluation search on a dataset file.

        Args:
            dataset_path: Path to dataset JSON file.
            k: Maximum number of retrieved sources per question.
            save_directory: Directory path to save search results.
            processed_dir: Path to directory containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
        """
        SearchDatasetCommand().cmd(
            dataset_path, k, save_directory, processed_dir, mode
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Evaluate retrieval performance metrics against ground truth.

        Args:
            student_search_results_path: Path to student search results JSON.
            dataset_path: Path to ground-truth dataset JSON file.
        """
        EvaluateCommand().cmd(student_search_results_path, dataset_path)

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        mode: str = "bm25",
    ) -> None:
        """Retrieve context and generate an answer for a user query.

        Args:
            query: Query question text string.
            k: Maximum number of source spans to retrieve.
            processed_dir: Path to directory containing index files.
            mode: Retrieval strategy mode ('bm25' or 'hybrid').
        """
        from .answer import AnswerCommand

        AnswerCommand().cmd(query, k, processed_dir, mode)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a precomputed batch search results file.

        Args:
            student_search_results_path: Path to student search results JSON.
            save_directory: Directory path to save answer results.
        """
        from .answer import AnswerDatasetCommand

        AnswerDatasetCommand().cmd(
            student_search_results_path, save_directory
        )

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        processed_dir: str = "data/processed",
    ) -> None:
        """Launch the FastAPI HTTP server for search and answer APIs.

        Args:
            host: Network interface address to bind web server to.
            port: Network port number to listen on.
            processed_dir: Path to directory containing index files.
        """
        ServeCommand().cmd(host, port, processed_dir)


__all__: List[str] = ["Cli"]
