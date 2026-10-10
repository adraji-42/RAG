import math
import heapq
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from .tokenizer import BM25Tokenizer


class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.__k1: float = k1
        self.__b: float = b
        self.__corpus_size: int = 0
        self.__avg_doc_len: float = 0.0
        self.__doc_lengths: List[int] = []
        self.__idf: Dict[str, float] = {}
        self.__inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)

    @property
    def corpus_size(self) -> int:
        return self.__corpus_size

    @property
    def avg_doc_len(self) -> float:
        return self.__avg_doc_len

    def fit(self, corpus: List[str]) -> None:
        self.__corpus_size = len(corpus)
        self.__doc_lengths = []
        self.__inverted_index = defaultdict(dict)
        for doc_idx, doc in enumerate(corpus):
            tokens: List[str] = BM25Tokenizer.tokenize(doc)
            self.__doc_lengths.append(len(tokens))
            counts: Counter[str] = Counter(tokens)
            for term, freq in counts.items():
                self.__inverted_index[term][doc_idx] = freq
        self.__avg_doc_len = (
            sum(self.__doc_lengths) / self.__corpus_size
            if self.__corpus_size > 0
            else 0.0
        )
        n: float = float(self.__corpus_size)
        self.__idf = {
            term: math.log(
                1.0 + (n - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for term, postings in self.__inverted_index.items()
        }

    def get_scores(self, query: str) -> List[float]:
        scores: List[float] = [0.0] * self.__corpus_size
        b_factor: float = (
            (self.__k1 * self.__b) / self.__avg_doc_len
            if self.__avg_doc_len > 0.0
            else 0.0
        )
        k1_base: float = self.__k1 * (1.0 - self.__b)
        for token in set(BM25Tokenizer.tokenize(query)):
            if token not in self.__idf:
                continue
            idf: float = self.__idf[token]
            for doc_idx, freq in self.__inverted_index[token].items():
                num: float = idf * (freq * (self.__k1 + 1.0))
                denom: float = (
                    freq + k1_base + b_factor * self.__doc_lengths[doc_idx]
                )
                scores[doc_idx] += num / denom
        return scores

    def search(
        self, query: str, k: int = 5
    ) -> List[Tuple[int, float]]:
        return heapq.nlargest(
            k, enumerate(self.get_scores(query)), key=lambda x: x[1]
        )
