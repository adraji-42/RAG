"""Data models and schemas for retrieval and evaluation."""

import uuid
from typing import List, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent a character-indexed span in a source file.

    Attributes:
        file_path: Path to the referenced source document.
        first_character_index: 0-indexed start character offset.
        last_character_index: 0-indexed end character offset.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a query with an automatically assigned identifier.

    Attributes:
        question_id: Unique question identifier string.
        question: Query question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question paired with ground-truth sources and answer.

    Attributes:
        sources: List of ground-truth minimal source spans.
        answer: Reference answer string.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of answered or unanswered RAG questions.

    Attributes:
        rag_questions: List of question instances in the dataset.
    """

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Represent top retrieved sources for a question.

    Attributes:
        question_id: Unique question identifier string.
        question: Query question text.
        retrieved_sources: List of retrieved minimal source spans.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources and generated answer for a question.

    Attributes:
        answer: Generated answer text string.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Represent batch search results for evaluation across questions.

    Attributes:
        search_results: List of search results per question.
        k: Maximum number of retrieved sources per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent batch search results and generated answers.

    Attributes:
        search_results: List of answer results per question.
        k: Maximum number of retrieved sources per question.
    """

    search_results: List[MinimalAnswer]
    k: int
