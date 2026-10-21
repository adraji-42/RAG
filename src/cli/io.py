import os
from sys import stderr
from typing import Optional

from pydantic import BaseModel, ValidationError

from ..models import RagDataset, StudentSearchResults


class CliIO:

    @staticmethod
    def load_dataset(path: str) -> Optional[RagDataset]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return RagDataset.model_validate_json(f.read())
        except (ValidationError, OSError):
            print(f"Failed to read dataset file: {path}", file=stderr)
            return None

    @staticmethod
    def load_student_results(path: str) -> Optional[StudentSearchResults]:
        if not os.path.isfile(path):
            print(f"File not found: {path}", file=stderr)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return StudentSearchResults.model_validate_json(f.read())
        except (ValidationError, OSError):
            print(f"Invalid student search results file: {path}", file=stderr)
            return None

    @staticmethod
    def load_eval_dataset(path: str) -> Optional[RagDataset]:
        if not os.path.isfile(path):
            print(f"File not found: {path}", file=stderr)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return RagDataset.model_validate_json(f.read())
        except (ValidationError, OSError):
            print(f"Invalid dataset file: {path}", file=stderr)
            return None

    @staticmethod
    def save_result_file(
        model: BaseModel,
        save_dir: str,
        filename: str,
        message_prefix: str,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        out_path: str = os.path.join(save_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
        print(f"Saved {message_prefix} to {out_path}")
