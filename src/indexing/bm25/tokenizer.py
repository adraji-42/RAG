import re
from typing import List, Pattern

_TOKEN_RE: Pattern[str] = re.compile(r"\w+")
_TRANS1_RE: Pattern[str] = re.compile(r"([a-z0-9])([A-Z])")
_TRANS2_RE: Pattern[str] = re.compile(r"([A-Z]+)([A-Z][a-z])")
_SPLIT_RE: Pattern[str] = re.compile(r"[_ ]+")


class BM25Tokenizer:

    @staticmethod
    def tokenize(text: str) -> List[str]:
        raw_tokens: List[str] = _TOKEN_RE.findall(text)
        result: List[str] = []
        for raw in raw_tokens:
            s1: str = _TRANS1_RE.sub(r"\1 \2", raw)
            s2: str = _TRANS2_RE.sub(r"\1 \2", s1)
            parts: List[str] = [
                p.lower() for p in _SPLIT_RE.split(s2) if p
            ]
            lower_raw: str = raw.lower()
            candidates: List[str] = [lower_raw]
            for p in parts:
                if p != lower_raw:
                    candidates.append(p)
            for cand in candidates:
                if len(cand) >= 2 and cand.strip("_"):
                    result.append(cand)
        return result
