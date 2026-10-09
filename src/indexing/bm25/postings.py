from typing import Dict, List


class InvertedIndex:

    def __init__(self) -> None:
        self.__index: Dict[str, Dict[int, int]] = {}

    def add_term(self, term: str, doc_idx: int, freq: int) -> None:
        if term not in self.__index:
            self.__index[term] = {}
        self.__index[term][doc_idx] = freq

    def get_postings(self, term: str) -> Dict[int, int]:
        return self.__index.get(term, {})

    def doc_freq(self, term: str) -> int:
        return len(self.__index.get(term, {}))

    @property
    def terms(self) -> List[str]:
        return list(self.__index.keys())
