import re
from typing import List, Pattern

_TOKEN_RE: Pattern[str] = re.compile(r"\w+(?:[-.:/]\w+)*")
_TRANS1_RE: Pattern[str] = re.compile(r"([a-z0-9])([A-Z])")
_TRANS2_RE: Pattern[str] = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NUM_TRANS1_RE: Pattern[str] = re.compile(r"([a-zA-Z]{2,})([0-9]+)")
_NUM_TRANS2_RE: Pattern[str] = re.compile(r"([0-9]+)([a-zA-Z]{2,})")
_SPLIT_RE: Pattern[str] = re.compile(r"[-_ .:/]+")


class BM25Tokenizer:

    @staticmethod
    def tokenize(text: str) -> List[str]:
        result: List[str] = []

        for raw in _TOKEN_RE.findall(text):
            s: str = _NUM_TRANS2_RE.sub(
                r"\1 \2", _NUM_TRANS1_RE.sub(
                    r"\1 \2", _TRANS2_RE.sub(
                        r"\1 \2", _TRANS1_RE.sub(r"\1 \2", raw)
                    )
                )
            )
            parts: List[str] = [p.lower() for p in _SPLIT_RE.split(s) if p]
            lower_raw: str = raw.lower()
            candidates: List[str] = [lower_raw]
            for p in parts:
                if p != lower_raw:
                    candidates.append(p)
            result.extend([
                cand for cand in candidates
                if len(cand) >= 2 and not cand.isdigit()
            ])

        return result
