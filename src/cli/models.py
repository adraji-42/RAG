"""Pydantic configuration and validation models for CLI commands.""""""Pydantic configuration and validation models for CLI commands."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


RetrievalMode = Literal["bm25", "hybrid"]


class StrictBaseModel(BaseModel):
    """Base model enforcing strict type validation across all subclasses."""

    model_config = ConfigDict(strict=True)


class IndexOptions(StrictBaseModel):
    """Configuration options for corpus indexing.

    Attributes:
        max_chunk_size: Maximum allowed characters per chunk.
        raw_dir: Corpus directory path containing raw documents.
        processed_dir: Output directory path for processed index files.
        force: Force full corpus re-indexing from scratch.
        embed: Whether to compute dense vector embeddings.
    """

    max_chunk_size: int = Field(default=2000, gt=500)
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    force: bool = False
    embed: bool = False


class SearchOptions(StrictBaseModel):
    """Configuration options for single query search.

    Attributes:
        query: Query question text string.
        k: Maximum number of source spans to retrieve.
        processed_dir: Path to directory containing index files.
        mode: Retrieval strategy mode ('bm25' or 'hybrid').
    """

    query: str
    k: int = Field(default=5, gt=0)
    processed_dir: str = "data/processed"
    mode: RetrievalMode = "bm25"

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is a non-empty string.

        Args:
            v: Input query string to validate.

        Returns:
            Validated non-empty query string.
        """
        if not v.strip():
            raise ValueError("Query must be a non-empty string")
        return v


class SearchDatasetOptions(StrictBaseModel):
    """Configuration options for batch dataset query retrieval.

    Attributes:
        dataset_path: Path to dataset JSON file.
        k: Maximum number of retrieved sources per question.
        save_directory: Directory path to save search results.
        processed_dir: Path to directory containing index files.
        mode: Retrieval strategy mode ('bm25' or 'hybrid').
    """

    dataset_path: str
    k: int = Field(default=5, gt=0)
    save_directory: str = "data/output/search_results"
    processed_dir: str = "data/processed"
    mode: RetrievalMode = "bm25"


class EvaluateOptions(StrictBaseModel):
    """Configuration options for retrieval evaluation against ground truth.

    Attributes:
        student_search_results_path: Path to student search results JSON.
        dataset_path: Path to ground-truth dataset JSON file.
    """

    student_search_results_path: str
    dataset_path: str


class AnswerOptions(StrictBaseModel):
    """Configuration options for single query question answering.

    Attributes:
        query: Query question text string.
        k: Maximum number of source spans to retrieve.
        processed_dir: Path to directory containing index files.
        mode: Retrieval strategy mode ('bm25' or 'hybrid').
    """

    query: str
    k: int = Field(default=5, gt=0)
    processed_dir: str = "data/processed"
    mode: RetrievalMode = "bm25"

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is a non-empty string.

        Args:
            v: Input query string to validate.

        Returns:
            Validated non-empty query string.
        """
        if not v.strip():
            raise ValueError("Query must be a non-empty string")
        return v


class AnswerDatasetOptions(StrictBaseModel):
    """Configuration options for batch answer generation over search results.

    Attributes:
        student_search_results_path: Path to student search results JSON.
        save_directory: Directory path to save answer results.
    """

    student_search_results_path: str
    save_directory: str = "data/output/search_results_and_answer"


class ServeOptions(StrictBaseModel):
    """Configuration options for launching FastAPI web server.

    Attributes:
        host: Network interface address to bind web server to.
        port: Network port number on which to listen.
        processed_dir: Path to directory containing index files.
    """

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    processed_dir: str = "data/processed"
