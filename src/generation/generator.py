import os
from typing import Any, Dict, List, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models import MinimalSource
from .prompts import PromptBuilder


class AnswerGenerator:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 300,
    ) -> None:
        self.__max_new_tokens: int = max_new_tokens
        torch.set_num_threads(os.cpu_count() or 4)
        self.__tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.__model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        cast(torch.nn.Module, self.__model).eval()

    def __apply_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            return cast(
                str,
                self.__tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
            )
        except TypeError:
            return cast(
                str,
                self.__tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )

    def __run_model(self, prompt: str) -> str:
        inputs = self.__tokenizer(prompt, return_tensors="pt")
        input_len: int = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids: torch.Tensor = cast(
                torch.Tensor,
                self.__model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.__max_new_tokens,
                    pad_token_id=self.__tokenizer.eos_token_id,
                ),
            )
        new_tokens = output_ids[0][input_len:]
        return cast(
            str,
            self.__tokenizer.decode(new_tokens, skip_special_tokens=True),
        )

    def generate(self, question: str, sources: List[MinimalSource]) -> str:
        if not question.strip() or not sources:
            return "Insufficient context to answer the question."
        messages = PromptBuilder.build_messages(question, sources)
        prompt: str = self.__apply_template(messages)
        raw: str = self.__run_model(prompt)
        return PromptBuilder.strip_thinking(raw)
