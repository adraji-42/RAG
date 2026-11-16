"""Autoregressive language model generator with chat templates and caching."""

import hashlib
from typing import Any, Dict, List, Optional, cast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import PromptBuilder
from ..models import MinimalSource
from ..retrieval.cache import QueryCache

_DEFAULT_CONTEXT_WINDOW: int = 32768


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
        self.__tokenizer: Any = None
        self.__model: Any = None
        self.__max_input_tokens: int = 0
        self.__cache: QueryCache = QueryCache(processed_dir)

    def __ensure_loaded(self) -> None:
        """Load model weights and tokenizer in bfloat16 evaluation mode."""
        if self.__tokenizer is not None and self.__model is not None:
            return
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        self.__model = AutoModelForCausalLM.from_pretrained(
            self.__model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        cast(torch.nn.Module, self.__model).eval()
        context_window = self.__context_window()
        if context_window - self.__max_new_tokens <= 0:
            threshold = int(context_window * 0.34)
            raise ValueError(
                f"Max new tokens must be less than {threshold}"
            )
        self.__max_input_tokens = context_window - self.__max_new_tokens

    def __context_window(self) -> int:
        """Resolve dynamic context window size from loaded model config.

        Returns:
            Maximum context window size in tokens supported by the model.
        """
        cfg: Any = getattr(self.__model, "config", None)
        cfg_limit: int = int(
            getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "max_sequence_length", None)
            or getattr(cfg, "seq_length", None)
            or 0
        )
        tok_val: Any = getattr(self.__tokenizer, "model_max_length", None)
        tok_limit: int = int(tok_val) if tok_val else 0
        limits = [x for x in (cfg_limit, tok_limit) if x > 0]
        return min(limits) if limits else _DEFAULT_CONTEXT_WINDOW

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
        max_in: int = self.__max_input_tokens
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_in,
        )
        input_len: int = inputs["input_ids"].shape[1]
        pad_id = (
            tok.pad_token_id
            if tok.pad_token_id is not None
            else (
                tok.eos_token_id[0]
                if isinstance(tok.eos_token_id, list)
                else tok.eos_token_id
            )
        )
        with torch.inference_mode():
            output_ids: torch.Tensor = cast(
                torch.Tensor,
                mdl.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.__max_new_tokens,
                    pad_token_id=pad_id,
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
        parts: List[str] = [
            self.__model_name,
            str(self.__max_new_tokens),
            question.strip(),
        ]
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
        try:
            messages = PromptBuilder.build_messages(
                question,
                sources,
                tokenizer=self.__tokenizer,
                max_input_tokens=self.__max_input_tokens,
            )
        except ValueError as e:
            return str(e)
        prompt: str = self.__apply_template(messages)
        raw: str = self.__run_model(prompt)
        answer: str = PromptBuilder.strip_thinking(raw)
        if not answer.strip():
            return "The model failed to generate an answer."
        self.__cache.set_answer(cache_key, answer)
        return answer
