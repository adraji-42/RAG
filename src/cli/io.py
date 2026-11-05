"""CLI input/output helpers for reading and persisting JSON datasets."""

import os
from sys import stderr
from typing import Optional

from pydantic import BaseModel, ValidationError

from ..models import RagDataset, StudentSearchResults


class CliIO:
    """Handle JSON dataset loading, validation, and serialization for CLI."""

    @staticmethod
    def load_dataset(path: str) -> Optional[RagDataset]:
        """Load and validate a RAG dataset JSON file.

        Args:
            path: Path to dataset JSON file.

        Returns:
            Validated RagDataset instance or None on error.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return RagDataset.model_validate_json(f.read())
        except (ValidationError, OSError) as e:
            print(
                f"Failed to read dataset file '{path}': {e}", file=stderr
            )
            return None

    @staticmethod
    def load_student_results(path: str) -> Optional[StudentSearchResults]:
        """Load and validate student search results JSON file.

        Args:
            path: Path to student search results JSON file.

        Returns:
            Validated StudentSearchResults instance or None on error.
        """
        if not os.path.isfile(path):
            print(f"File not found: {path}", file=stderr)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return StudentSearchResults.model_validate_json(f.read())
        except (ValidationError, OSError) as e:
            print(
                f"Invalid student search results file '{path}': {e}",
                file=stderr,
            )
            return None

    @staticmethod
    def load_eval_dataset(path: str) -> Optional[RagDataset]:
        """Load and validate a ground-truth evaluation dataset JSON file.

        Args:
            path: Path to evaluation dataset JSON file.

        Returns:
            Validated RagDataset instance or None on error.
        """
        if not os.path.isfile(path):
            print(f"File not found: {path}", file=stderr)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return RagDataset.model_validate_json(f.read())
        except (ValidationError, OSError) as e:
            print(
                f"Invalid dataset file '{path}': {e}", file=stderr
            )
            return None

    @staticmethod
    def save_result_file(
        model: BaseModel,
        save_dir: str,
        filename: str,
        message_prefix: str,
    ) -> None:
        """Save a Pydantic model serialized as formatted JSON to disk.

        Args:
            model: Pydantic model instance to serialize.
            save_dir: Destination directory path.
            filename: Output file name.
            message_prefix: Log message descriptor prefix.
        """
        os.makedirs(save_dir, exist_ok=True)
        out_path: str = os.path.join(save_dir, filename)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(model.model_dump_json(indent=2))
            print(f"Saved {message_prefix} to {out_path}")
        except OSError as e:
            print(f"Failed to save {message_prefix}: {e}", file=stderr)
