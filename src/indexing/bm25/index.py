from typing import Dict, List

from .postings import InvertedIndex
from .scorer import BM25Scorer
from .stats import CorpusStats
from .tokenizer import BM25Tokenizer


class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.__stats: CorpusStats = CorpusStats()
        self.__postings: InvertedIndex = InvertedIndex()
        self.__scorer: BM25Scorer = BM25Scorer(k1, b)
        self.__idf: Dict[str, float] = {}

    def fit(self, corpus: List[str]) -> None:
        for idx, doc in enumerate(corpus):
            tokens: List[str] = BM25Tokenizer.tokenize(doc)
            self.__stats.add_doc(len(tokens))
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, freq in counts.items():
                self.__postings.add_term(t, idx, freq)
        self.__stats.finalize()
        c_size: int = self.__stats.corpus_size
        self.__idf = {
            t: self.__scorer.compute_idf(self.__postings.doc_freq(t), c_size)
            for t in self.__postings.terms
        }

    def get_scores(self, query: str) -> List[float]:
        scores: List[float] = [0.0] * self.__stats.corpus_size
        doc_lens: List[int] = self.__stats.doc_lengths
        avg_len: float = self.__stats.avg_doc_len
        for t in BM25Tokenizer.tokenize(query):
            if t not in self.__idf:
                continue
            idf: float = self.__idf[t]
            for idx, freq in self.__postings.get_postings(t).items():
                scores[idx] += self.__scorer.score_term(
                    freq, doc_lens[idx], avg_len, idf
                )
        return scores
