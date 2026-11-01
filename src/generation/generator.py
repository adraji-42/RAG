import hashlib
from typing import Any, Dict, List, Optional, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import PromptBuilder
from ..models import MinimalSource
from ..retrieval.cache import QueryCache


class AnswerGenerator:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 300,
        processed_dir: str = "data/processed",
    ) -> None:
        self.__model_name: str = model_name
        self.__max_new_tokens: int = max_new_tokens
        self.__cache: QueryCache = QueryCache(processed_dir)
        self.__tokenizer: Any = None
        self.__model: Any = None

    def __ensure_loaded(self) -> None:
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
        parts: List[str] = [question.strip()]
        for s in sources:
            parts.append(
                f"{s.file_path}:{s.first_character_index}"
                f":{s.last_character_index}"
            )
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def generate(self, question: str, sources: List[MinimalSource]) -> str:
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
