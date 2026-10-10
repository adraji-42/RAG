import math
import heapq
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from .tokenizer import BM25Tokenizer


class BM25Index:

    def __init__(
        self, corpus: List[str], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.__doc_lengths: List[int] = []
        self.__inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        for doc_idx, doc in enumerate(corpus):
            tokens: List[str] = BM25Tokenizer.tokenize(doc)
            self.__doc_lengths.append(len(tokens))
            counts: Counter[str] = Counter(tokens)
            for term, freq in counts.items():
                self.__inverted_index[term][doc_idx] = freq
        self.__avg_doc_len: float = (
            sum(self.__doc_lengths) / len(self.__doc_lengths)
            if self.__doc_lengths
            else 0.0
        )
        n: float = float(len(self.__doc_lengths))
        self.__idf: Dict[str, float] = {
            term: math.log(
                (n - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for term, postings in self.__inverted_index.items()
        }
        self.__k1_base: float = k1 * (1 - b)
        self.__b_factor: float = (
            (k1 * b) / self.__avg_doc_len if self.__avg_doc_len > 0.0 else 0.0
        )
        self.__k1_plus_1: float = k1 + 1.0

    @property
    def corpus_size(self) -> int:
        return len(self.__doc_lengths)

    @property
    def avg_doc_len(self) -> float:
        return self.__avg_doc_len

    def get_scores(self, query: str) -> List[float]:
        scores: List[float] = [0.0] * len(self.__doc_lengths)
        for token in set(BM25Tokenizer.tokenize(query)):
            if token not in self.__idf:
                continue
            idf: float = self.__idf[token]
            for doc_idx, freq in self.__inverted_index[token].items():
                scores[doc_idx] += (
                    (idf * freq * self.__k1_plus_1)
                    / (
                        freq
                        + self.__k1_base
                        + self.__b_factor * self.__doc_lengths[doc_idx]
                    )
                )
        return scores

    def search(
        self, query: str, k: int = 5
    ) -> List[Tuple[int, float]]:
        return heapq.nlargest(
            k, enumerate(self.get_scores(query)), key=lambda x: x[1]
        )
