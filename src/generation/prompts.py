"""Prompt construction and context formatting for technical answering."""

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
    """Format technical contexts, chat messages, and strip think tags."""

    _THINK_RE: Pattern[str] = re.compile(r"<think>.*?</think>", re.DOTALL)

    @staticmethod
    def __read_snippet(s: MinimalSource) -> Optional[str]:
        """Read snippet text from disk within character index bounds.

        Args:
            s: MinimalSource metadata identifying the span.

        Returns:
            Extracted text snippet string, or None on read failure.
        """
        try:
            with open(
                s.file_path, "r", encoding="utf-8", errors="replace"
            ) as f:
                content: str = f.read()
            return content[
                s.first_character_index:s.last_character_index
            ].strip()
        except OSError:
            return None

    @staticmethod
    def build_context(
        sources: List[MinimalSource], max_chars: int = 6000
    ) -> str:
        """Assemble source snippets into context up to max characters.

        Args:
            sources: List of MinimalSource objects to include.
            max_chars: Maximum character budget for context.

        Returns:
            Formatted context block string from snippets.
        """
        total: int = 0
        parts: List[str] = []
        for s in sources:
            snippet: Optional[str] = PromptBuilder.__read_snippet(s)
            if snippet is None:
                continue
            block: str = f"[{s.file_path}]\n{snippet}"
            blen: int = len(block)
            if total + blen > max_chars:
                break
            parts.append(block)
            total += blen
        return "\n\n".join(parts)

    @staticmethod
    def build_messages(
        question: str, sources: List[MinimalSource]
    ) -> List[Dict[str, str]]:
        """Construct system and user chat message dictionaries.

        Args:
            question: Query question text string.
            sources: List of retrieved MinimalSource objects.

        Returns:
            List of system and user chat message dictionaries.
        """
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
        """Remove internal reasoning blocks delimited by think tags.

        Args:
            text: Raw model output text string.

        Returns:
            Cleaned answer string with reasoning blocks removed.
        """
        return cls._THINK_RE.sub("", text).strip()
