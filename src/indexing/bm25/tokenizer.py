import re


class BM25Tokenizer:

    @staticmethod
    def tokenize(text: str) -> list[str]:
        raw_tokens: list[str] = re.findall(r"\w+", text)
        result: list[str] = []
        for raw in raw_tokens:
            s1: str = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
            s2: str = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s1)
            parts: list[str] = [
                p.lower() for p in re.split(r"[_ ]+", s2) if p
            ]
            lower_raw: str = raw.lower()
            candidates: list[str] = [lower_raw]
            for p in parts:
                if p != lower_raw:
                    candidates.append(p)
            for cand in candidates:
                if len(cand) >= 2 and cand.strip("_"):
                    result.append(cand)
        return result
