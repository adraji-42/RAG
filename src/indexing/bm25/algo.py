import os
import math
import pickle
from typing import Dict, List, Self

from .tokenizer import BM25Tokenizer


class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.__k1: float = k1
        self.__b: float = b
        self.__corpus_size: int = 0
        self.__avg_doc_len: float = 0.0
        self.__doc_lengths: List[int] = []
        self.__idf: Dict[str, float] = {}
        self.__inverted_index: Dict[str, Dict[int, int]] = {}

    def fit(self, corpus: List[str]) -> None:
        self.__corpus_size = len(corpus)
        if self.__corpus_size == 0:
            self.__avg_doc_len = 0.0
            self.__doc_lengths = []
            self.__idf = {}
            self.__inverted_index = {}
            return

        self.__doc_lengths: List[int] = []
        self.__inverted_index: Dict[str, Dict[int, int]] = {}
        total_tokens: int = 0

        for doc_idx, doc in enumerate(corpus):
            tokens: List[str] = BM25Tokenizer.tokenize(doc)
            doc_len: int = len(tokens)
            self.__doc_lengths.append(doc_len)
            total_tokens += doc_len

            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            for token, freq in counts.items():
                if token not in self.__inverted_index:
                    self.__inverted_index[token] = {}
                self.__inverted_index[token][doc_idx] = freq

        self.__avg_doc_len: float = total_tokens / self.__corpus_size

        self.__idf: Dict[str, float] = {}
        n_docs: float = float(self.__corpus_size)
        for term, postings in self.__inverted_index.items():
            n_q: float = float(len(postings))
            self.__idf[term] = math.log(
                1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5)
            )

    def get_scores(self, query: str) -> List[float]:
        query_tokens: List[str] = BM25Tokenizer.tokenize(query)
        scores: List[float] = [0.0] * self.__corpus_size
        for token in query_tokens:
            if token not in self.__inverted_index:
                continue
            idf_val: float = self.__idf[token]
            postings: Dict[int, int] = self.__inverted_index[token]
            for doc_idx, freq in postings.items():
                len_norm: float = (
                    self.__doc_lengths[doc_idx] / self.__avg_doc_len
                    if self.__avg_doc_len > 0.0
                    else 0.0
                )
                denom: float = freq + self.__k1 * (
                    1.0 - self.__b + self.__b * len_norm
                )
                if denom > 0.0:
                    scores[doc_idx] += idf_val * (
                        freq * (self.__k1 + 1.0)
                    ) / denom
        return scores

    def scores(self, query: str) -> List[float]:
        return self.get_scores(query)

    def save(self, file_path: str) -> None:
        parent: str = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str) -> Self:
        with open(file_path, "rb") as f:
            index: Self = pickle.load(f)
            return index
