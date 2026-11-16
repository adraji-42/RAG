"""Prompt construction and context formatting for technical answering."""

import re
from typing import Any, Dict, List, Optional, Pattern, cast
from ..models import MinimalSource


SYSTEM_PROMPT: str = (
    "You are a precise technical assistant. "
    "Answer the question accurately and concisely using ONLY the provided "
    "code or documentation snippets. "
    "Do not hallucinate or add information not found in the snippets."
)


class PromptBuilder:
    """Format technical contexts, chat messages, and strip think tags."""

    _THINK_RE: Pattern[str] = re.compile(
        r"<think>.*?(?:</think>|$)", re.DOTALL
    )

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
                s.file_path,
                "r",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as f:
                content: str = f.read()
            return content[
                s.first_character_index:s.last_character_index
            ].strip()
        except OSError:
            return None

    @staticmethod
    def __count_tokens(tokenizer: Any, text: str) -> int:
        """Count token length of text using the model tokenizer.

        Args:
            tokenizer: Pretrained Hugging Face tokenizer instance.
            text: Input text string to tokenize.

        Returns:
            Number of tokens in the encoded text.
        """
        ids: List[int] = cast(
            List[int], tokenizer.encode(text, add_special_tokens=False)
        )
        return len(ids)

    @classmethod
    def __base_tokens(cls, tokenizer: Any, question: str) -> int:
        """Count exact tokens of system prompt, query, and template.

        Args:
            tokenizer: Pretrained Hugging Face tokenizer instance.
            question: User query text string.

        Returns:
            Number of tokens consumed before adding context chunks.
        """
        empty_user: str = f"Context:\n\n\nQuestion: {question}"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": empty_user},
        ]
        kwargs: Dict[str, bool] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            rendered = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        return cls.__count_tokens(tokenizer, cast(str, rendered))

    @classmethod
    def build_context(
        cls,
        sources: List[MinimalSource],
        tokenizer: Any,
        max_context_tokens: int,
    ) -> str:
        """Assemble source snippets into context within token budget.

        Args:
            sources: List of MinimalSource objects to include.
            tokenizer: Pretrained Hugging Face tokenizer for token counting.
            max_context_tokens: Maximum token budget allocated for chunks.

        Returns:
            Formatted context block string from snippets.

        Raises:
            ValueError: If not even one source fits within the token budget.
        """
        used: int = 0
        parts: List[str] = []
        has_snippet: bool = False
        for s in sources:
            snippet: Optional[str] = cls.__read_snippet(s)
            if not snippet:
                continue
            has_snippet = True
            block: str = f"[{s.file_path}]\n{snippet}"
            candidate: str = f"\n\n{block}" if parts else block
            cost: int = cls.__count_tokens(tokenizer, candidate)
            if used + cost > max_context_tokens:
                break
            parts.append(block)
            used += cost
        if not parts:
            if not has_snippet:
                raise ValueError(
                    "No valid source snippets available to build context."
                )
            raise ValueError(
                "Prompt is too long: cannot fit even one "
                "retrieved source within the context window."
            )
        return "\n\n".join(parts)

    @classmethod
    def build_messages(
        cls,
        question: str,
        sources: List[MinimalSource],
        tokenizer: Any,
        max_input_tokens: int,
    ) -> List[Dict[str, str]]:
        """Build system and user chat messages within token budget.

        Args:
            question: Query question text string.
            sources: List of retrieved MinimalSource objects.
            tokenizer: Pretrained Hugging Face tokenizer for token counting.
            max_input_tokens: Maximum total input tokens allowed for prompt.

        Returns:
            List of system and user chat message dictionaries.

        Raises:
            ValueError: If no retrieved source fits within the token budget.
        """
        base_len: int = cls.__base_tokens(tokenizer, question)
        budget: int = max_input_tokens - base_len
        context: str = cls.build_context(sources, tokenizer, budget)
        user_content: str = f"Context:\n{context}\n\nQuestion: {question}"
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
