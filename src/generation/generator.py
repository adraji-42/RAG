"""Autoregressive language model generator with chat templates and caching."""

import hashlib
from typing import Any, Dict, List, Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import PromptBuilder
from ..models import MinimalSource
from ..retrieval.cache import QueryCache


class AnswerGenerator:
    """Generate grounded answers using an autoregressive language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 300,
        processed_dir: str = "data/processed",
    ) -> None:
        """Initialize the answer generator with model name and token limits.

        Args:
            model_name: Pretrained causal language model name or path.
            max_new_tokens: Maximum number of new tokens to generate.
            processed_dir: Path to directory containing answer cache.
        """
        self.__model_name: str = model_name
        self.__max_new_tokens: int = max_new_tokens
        self.__cache: QueryCache = QueryCache(processed_dir)
        self.__tokenizer: Any = None
        self.__model: Any = None

    def __ensure_loaded(self) -> None:
        """Load model weights and tokenizer in bfloat16 evaluation mode."""
        if self.__tokenizer is not None:
            return
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        self.__model = AutoModelForCausalLM.from_pretrained(
            self.__model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        cast(torch.nn.Module, self.__model).eval()

    def __apply_template(self, messages: List[Any]) -> str:
        """Render chat prompt template without thinking blocks.

        Args:
            messages: List of chat message dictionaries.

        Returns:
            Formatted prompt text string for generation.
        """
        tok: Any = self.__tokenizer
        kwargs: Dict[str, bool] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            res = tok.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            res = tok.apply_chat_template(messages, **kwargs)
        return cast(str, res)

    def __run_model(self, prompt: str) -> str:
        """Generate tokens greedily using loaded causal language model.

        Args:
            prompt: Formatted prompt text string.

        Returns:
            Decoded generated answer string.
        """
        tok: Any = self.__tokenizer
        mdl: Any = self.__model
        inputs = tok(prompt, return_tensors="pt")
        input_len: int = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids: torch.Tensor = cast(
                torch.Tensor,
                mdl.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.__max_new_tokens,
                    pad_token_id=tok.eos_token_id,
                ),
            )
        new_tokens = output_ids[0][input_len:]
        return cast(str, tok.decode(new_tokens, skip_special_tokens=True))

    def __build_cache_key(
        self, question: str, sources: List[MinimalSource]
    ) -> str:
        """Compute SHA-256 hash from question and retrieved source bounds.

        Args:
            question: Query question string.
            sources: List of retrieved MinimalSource objects.

        Returns:
            SHA-256 hexadecimal cache key string.
        """
        parts: List[str] = [question.strip()]
        for s in sources:
            parts.append(
                f"{s.file_path}:{s.first_character_index}"
                f":{s.last_character_index}"
            )
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def generate(self, question: str, sources: List[MinimalSource]) -> str:
        """Generate answer from retrieved sources or return cached result.

        Args:
            question: Query question string.
            sources: List of retrieved MinimalSource objects.

        Returns:
            Generated or cached answer text string.
        """
        if not question.strip() or not sources:
            return "Insufficient context to answer the question."
        cache_key: str = self.__build_cache_key(question, sources)
        cached: Optional[str] = self.__cache.get_answer(cache_key)
        if cached is not None:
            return cached
        self.__ensure_loaded()
        messages = PromptBuilder.build_messages(question, sources)
        prompt: str = self.__apply_template(messages)
        raw: str = self.__run_model(prompt)
        answer: str = PromptBuilder.strip_thinking(raw)
        self.__cache.set_answer(cache_key, answer)
        return answer
