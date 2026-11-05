"""FastAPI web service exposing search and answer REST endpoints."""

import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from ..generation import AnswerGenerator
from ..retrieval import BaseRetriever, BM25Retriever, HybridRetriever
from ..models import MinimalAnswer, MinimalSearchResults, MinimalSource


class QueryRequest(BaseModel):
    """Request schema for search and answer API endpoints.

    Attributes:
        query: Query question text string.
        k: Maximum number of source spans to retrieve.
        mode: Retrieval strategy mode ('bm25' or 'hybrid').
    """

    query: str
    k: int = Field(default=5, gt=0)
    mode: str = "bm25"


class RagApiService:
    """FastAPI service exposing retrieval and answering endpoints."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Initialize the API service and register application routes.

        Args:
            processed_dir: Path to directory containing index artifacts.
        """
        self.__processed_dir: str = processed_dir
        self.__bm25: Optional[BM25Retriever] = None
        self.__hybrid: Optional[HybridRetriever] = None
        self.__generator: Optional[AnswerGenerator] = None
        self.__app: FastAPI = FastAPI(title="RAG API")
        self.__register_routes()

    @property
    def app(self) -> FastAPI:
        """Return the underlying FastAPI application instance.

        Returns:
            FastAPI application instance with registered routes.
        """
        return self.__app

    def __get_retriever(self, mode: str) -> BaseRetriever:
        """Return or instantiate retriever corresponding to requested mode.

        Args:
            mode: Retrieval strategy mode ('bm25' or 'hybrid').

        Returns:
            Configured BaseRetriever instance.

        Raises:
            ValueError: If mode is not 'bm25' or 'hybrid'.
        """
        if mode not in ("bm25", "hybrid"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'bm25' or 'hybrid'."
            )
        if mode == "hybrid":
            if self.__hybrid is None:
                self.__hybrid = HybridRetriever(self.__processed_dir)
            return self.__hybrid
        if self.__bm25 is None:
            self.__bm25 = BM25Retriever(self.__processed_dir)
        return self.__bm25

    def __get_generator(self) -> AnswerGenerator:
        """Return or instantiate lazy-loaded answer generator.

        Returns:
            Configured AnswerGenerator instance.
        """
        if self.__generator is None:
            self.__generator = AnswerGenerator(
                processed_dir=self.__processed_dir
            )
        return self.__generator

    def __search(self, req: QueryRequest) -> MinimalSearchResults:
        """Handle query search requests and return retrieved sources.

        Args:
            req: QueryRequest containing search parameters.

        Returns:
            MinimalSearchResults containing retrieved sources.

        Raises:
            HTTPException: If query processing or retrieval fails.
        """
        try:
            sources: List[MinimalSource] = self.__get_retriever(
                req.mode
            ).retrieve(req.query, k=req.k)
            return MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=req.query,
                retrieved_sources=sources,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def __answer(self, req: QueryRequest) -> MinimalAnswer:
        """Handle query answer requests and return generated answers.

        Args:
            req: QueryRequest containing query and retrieval parameters.

        Returns:
            MinimalAnswer containing retrieved sources and answer.

        Raises:
            HTTPException: If answering or retrieval fails.
        """
        try:
            sources: List[MinimalSource] = self.__get_retriever(
                req.mode
            ).retrieve(req.query, k=req.k)
            ans: str = self.__get_generator().generate(req.query, sources)
            return MinimalAnswer(
                question_id=str(uuid.uuid4()),
                question=req.query,
                retrieved_sources=sources,
                answer=ans,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def __register_routes(self) -> None:
        """Register HTTP route handlers on the FastAPI application."""
        @self.__app.get("/health")
        def health() -> Dict[str, str]:
            """Return service health status.

            Returns:
                Dictionary with service operational status.
            """
            return {"status": "ok"}

        self.__app.post("/search")(self.__search)
        self.__app.post("/answer")(self.__answer)
