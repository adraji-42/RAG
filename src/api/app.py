import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from ..generation import AnswerGenerator
from ..retrieval import BaseRetriever, BM25Retriever, HybridRetriever
from ..models import MinimalAnswer, MinimalSearchResults, MinimalSource


class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=5, gt=0)
    mode: str = "bm25"


class RagApiService:
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.__processed_dir: str = processed_dir
        self.__bm25: Optional[BM25Retriever] = None
        self.__hybrid: Optional[HybridRetriever] = None
        self.__generator: Optional[AnswerGenerator] = None
        self.__app: FastAPI = FastAPI(title="RAG API")
        self.__register_routes()

    @property
    def app(self) -> FastAPI:
        return self.__app

    def __get_retriever(self, mode: str) -> BaseRetriever:
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
        if self.__generator is None:
            self.__generator = AnswerGenerator(
                processed_dir=self.__processed_dir
            )
        return self.__generator

    def __search(self, req: QueryRequest) -> MinimalSearchResults:
        try:
            sources: List[MinimalSource] = self.__get_retriever(
                req.mode
            ).retrieve(req.query, k=req.k)
            return MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=req.query,
                retrieved_sources=sources,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    def __answer(self, req: QueryRequest) -> MinimalAnswer:
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
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    def __register_routes(self) -> None:
        @self.__app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        self.__app.post("/search")(self.__search)
        self.__app.post("/answer")(self.__answer)
