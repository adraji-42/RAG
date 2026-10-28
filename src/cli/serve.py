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
        service: RagApiService = RagApiService(processed_dir)
        uvicorn.run(service.app, host=host, port=port)
