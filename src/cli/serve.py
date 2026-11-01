import sys

import uvicorn

from .base import BaseCommand
from ..api import RagApiService


class ServeCommand(BaseCommand):
    def cmd(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        processed_dir: str = "data/processed",
    ) -> None:
        try:
            service: RagApiService = RagApiService(processed_dir)
            uvicorn.run(service.app, host=host, port=port)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
