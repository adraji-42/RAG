from typing import Any

from pydantic import BaseModel, Field, field_validator


class IndexOptions(BaseModel):
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

    @field_validator("max_chunk_size", mode="before")
    @classmethod
    def validate_max_chunk_size(cls, v: Any) -> Any:
        """Validate max_chunk_size is an integer, rejecting boolean values."""
        if isinstance(v, bool):
            raise ValueError(
                "requires a positive integer, not a boolean flag"
            )
        return v


class SearchOptions(BaseModel):
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
    mode: str = "bm25"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is either bm25 or hybrid."""
        if v not in ("bm25", "hybrid"):
            raise ValueError("must be 'bm25' or 'hybrid'")
        return v

    @field_validator("k", mode="before")
    @classmethod
    def validate_k(cls, v: Any) -> Any:
        """Validate k is a positive integer, rejecting boolean values."""
        if isinstance(v, bool):
            raise ValueError(
                "requires a positive integer, not a boolean flag"
            )
        return v

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: Any) -> Any:
        """Validate query is a non-empty string."""
        if isinstance(v, bool) or not isinstance(v, str) or not v.strip():
            raise ValueError("Query must be a non-empty string")
        return v


class SearchDatasetOptions(BaseModel):
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
    mode: str = "bm25"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is either bm25 or hybrid."""
        if v not in ("bm25", "hybrid"):
            raise ValueError("must be 'bm25' or 'hybrid'")
        return v

    @field_validator("k", mode="before")
    @classmethod
    def validate_k(cls, v: Any) -> Any:
        """Validate k is a positive integer, rejecting boolean values."""
        if isinstance(v, bool):
            raise ValueError(
                "requires a positive integer, not a boolean flag"
            )
        return v


class EvaluateOptions(BaseModel):
    """Configuration options for retrieval evaluation against ground truth.

    Attributes:
        student_search_results_path: Path to student search results JSON.
        dataset_path: Path to ground-truth dataset JSON file.
    """

    student_search_results_path: str
    dataset_path: str


class AnswerOptions(BaseModel):
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
    mode: str = "bm25"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is either bm25 or hybrid."""
        if v not in ("bm25", "hybrid"):
            raise ValueError("must be 'bm25' or 'hybrid'")
        return v

    @field_validator("k", mode="before")
    @classmethod
    def validate_k(cls, v: Any) -> Any:
        """Validate k is a positive integer, rejecting boolean values."""
        if isinstance(v, bool):
            raise ValueError(
                "requires a positive integer, not a boolean flag"
            )
        return v

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: Any) -> Any:
        """Validate query is a non-empty string."""
        if isinstance(v, bool) or not isinstance(v, str) or not v.strip():
            raise ValueError("Query must be a non-empty string")
        return v


class AnswerDatasetOptions(BaseModel):
    """Configuration options for batch answer generation over search results.

    Attributes:
        student_search_results_path: Path to student search results JSON.
        save_directory: Directory path to save answer results.
    """

    student_search_results_path: str
    save_directory: str = "data/output/search_results_and_answer"


class ServeOptions(BaseModel):
    """Configuration options for launching FastAPI web server.

    Attributes:
        host: Network interface address to bind web server to.
        port: Network port number on which to listen.
        processed_dir: Path to directory containing index files.
    """

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    processed_dir: str = "data/processed"

    @field_validator("port", mode="before")
    @classmethod
    def validate_port(cls, v: Any) -> Any:
        """Validate port is a valid port integer, rejecting boolean values."""
        if isinstance(v, bool):
            raise ValueError(
                "requires a port number (1-65535), not a boolean flag"
            )
        return v
