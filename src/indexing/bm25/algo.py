import os
import math
import pickle
from typing import Self

from .tokenizer import BM25Tokenizer


class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1: float = k1
        self.b: float = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: list[int] = []
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, dict[int, int]] = {}

    def fit(self, corpus: list[str]) -> None:
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.doc_lengths = []
            self.idf = {}
            self.inverted_index = {}
            return

        self.doc_lengths = []
        self.inverted_index = {}
        total_tokens: int = 0

        for doc_idx, doc in enumerate(corpus):
            tokens: list[str] = BM25Tokenizer.tokenize(doc)
            doc_len: int = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            for token, freq in counts.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_idx] = freq

        self.avg_doc_len = total_tokens / self.corpus_size

        self.idf = {}
        n_docs: float = float(self.corpus_size)
        for term, postings in self.inverted_index.items():
            n_q: float = float(len(postings))
            self.idf[term] = math.log(
                1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5)
            )

    def get_scores(self, query: str) -> list[float]:
        query_tokens: list[str] = BM25Tokenizer.tokenize(query)
        scores: list[float] = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf_val: float = self.idf[token]
            postings: dict[int, int] = self.inverted_index[token]
            for doc_idx, freq in postings.items():
                len_norm: float = (
                    self.doc_lengths[doc_idx] / self.avg_doc_len
                    if self.avg_doc_len > 0.0
                    else 0.0
                )
                denom: float = freq + self.k1 * (
                    1.0 - self.b + self.b * len_norm
                )
                if denom > 0.0:
                    scores[doc_idx] += idf_val * (
                        freq * (self.k1 + 1.0)
                    ) / denom
        return scores

    def scores(self, query: str) -> list[float]:
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
