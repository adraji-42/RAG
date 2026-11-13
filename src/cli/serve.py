"""CLI command to start the FastAPI Uvicorn web server."""

import sys

import uvicorn
from pydantic import ValidationError

from .base import BaseCommand
from ..api import RagApiService
from .models import ServeOptions


class ServeCommand(BaseCommand):
    """Launch the API web server via command-line interface."""

    def cmd(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        processed_dir: str = "data/processed",
    ) -> None:
        """Instantiate the API service and run uvicorn server.

        Args:
            host: Network interface address to bind server to.
            port: Network port integer on which to listen.
            processed_dir: Path to directory containing index files.
        """
        try:
            options: ServeOptions = ServeOptions(
                host=host,
                port=port,
                processed_dir=processed_dir,
            )
            service: RagApiService = RagApiService(options.processed_dir)
            uvicorn.run(service.app, host=options.host, port=options.port)
        except ValidationError as e:
            print(self.format_validation_error(e), file=sys.stderr)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
