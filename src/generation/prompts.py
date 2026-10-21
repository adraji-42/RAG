import re
from typing import Dict, List, Optional, Pattern

from ..models import MinimalSource

SYSTEM_PROMPT: str = (
    "You are a precise technical assistant. "
    "Answer the question accurately and concisely using ONLY the provided "
    "code or documentation snippets. "
    "Do not hallucinate or add information not found in the snippets."
)


class PromptBuilder:

    _THINK_RE: Pattern[str] = re.compile(r"<think>.*?</think>", re.DOTALL)

    @staticmethod
    def __read_snippet(s: MinimalSource) -> Optional[str]:
        try:
            with open(
                s.file_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                content: str = f.read()
            return content[s.first_character_index:s.last_character_index]
        except OSError:
            return None

    @staticmethod
    def build_context(
        sources: List[MinimalSource], max_chars: int = 6000
    ) -> str:
        total: int = 0
        parts: List[str] = []
        for s in sources:
            snippet: Optional[str] = PromptBuilder.__read_snippet(s)
            if snippet is None:
                continue
            header: str = (
                f"[{s.file_path} "
                f"{s.first_character_index}:{s.last_character_index}]\n"
            )
            block: str = header + snippet + "\n"
            if total + len(block) > max_chars:
                rem: int = max_chars - total
                if rem > len(header):
                    parts.append(header + snippet[:rem - len(header)])
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)

    @staticmethod
    def build_messages(
        question: str, sources: List[MinimalSource]
    ) -> List[Dict[str, str]]:
        context: str = PromptBuilder.build_context(sources)
        user_content: str = (
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def strip_thinking(cls, text: str) -> str:
        return cls._THINK_RE.sub("", text).strip()
