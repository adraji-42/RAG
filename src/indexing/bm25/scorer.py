import math


class BM25Scorer:

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.__k1: float = k1
        self.__b: float = b

    def compute_idf(self, doc_freq: int, corpus_size: int) -> float:
        n_docs: float = float(corpus_size)
        n_q: float = float(doc_freq)
        return math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))

    def score_term(
        self, freq: int, doc_len: int, avg_doc_len: float, idf: float
    ) -> float:
        len_norm: float = (
            doc_len / avg_doc_len if avg_doc_len > 0.0 else 0.0
        )
        denom: float = freq + self.__k1 * (
            1.0 - self.__b + self.__b * len_norm
        )
        if denom <= 0.0:
            return 0.0
        return idf * (freq * (self.__k1 + 1.0)) / denom
