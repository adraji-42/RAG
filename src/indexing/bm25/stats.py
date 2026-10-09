from typing import List


class CorpusStats:

    def __init__(self) -> None:
        self.__corpus_size: int = 0
        self.__avg_doc_len: float = 0.0
        self.__doc_lengths: List[int] = []

    def add_doc(self, length: int) -> None:
        self.__doc_lengths.append(length)
        self.__corpus_size += 1

    def finalize(self) -> None:
        if self.__corpus_size > 0:
            self.__avg_doc_len = sum(self.__doc_lengths) / self.__corpus_size
        else:
            self.__avg_doc_len = 0.0

    @property
    def corpus_size(self) -> int:
        return self.__corpus_size

    @property
    def avg_doc_len(self) -> float:
        return self.__avg_doc_len

    @property
    def doc_lengths(self) -> List[int]:
        return self.__doc_lengths
