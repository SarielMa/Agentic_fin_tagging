from __future__ import annotations

import copy
import re
from typing import Any, Protocol

from .hf_generation import load_local_causal_lm, model_input_device, move_inputs_to_device


class Reranker(Protocol):
    def choose(self, entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        ...


class RetrievalTop1Reranker:
    """No-LLM baseline: choose the highest-scoring retrieved candidate."""

    def choose(self, entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        return candidates[0]["tag"] if candidates else ""


class LlamaReranker:
    """Local Hugging Face Llama reranker.

    This is intended for the GPU session. Set FINAI_LOCAL_FILES_ONLY=1 to force
    a fully cached/offline run.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_input_tokens: int = 12288,
        max_new_tokens: int = 48,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer, self.model, self.device = load_local_causal_lm(
            model_name,
            torch,
            AutoModelForCausalLM,
            AutoTokenizer,
            device=device,
        )

    def choose(self, entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return ""
        prompt = self._build_prompt(entity, entity_type, evidence, candidates)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = move_inputs_to_device(inputs, self.device or model_input_device(self.model))
        with self.torch.no_grad():
            generation_config = copy.deepcopy(getattr(self.model, "generation_config", None))
            if generation_config is not None:
                generation_config.do_sample = False
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = self._parse_tag(generated)
        return parsed or candidates[0]["tag"]

    @staticmethod
    def _build_prompt(entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        candidate_lines = "\n".join(
            f"{i}. {candidate['tag']} | {str(candidate['text'])[:160]}"
            for i, candidate in enumerate(candidates, start=1)
        )
        return (
            "You are a financial tagging assistant trained in US-GAAP taxonomy.\n"
            "Given a target entity, its XBRL type, local evidence, and candidate tags, "
            "select the single best US-GAAP tag.\n"
            "Return only JSON like {\"result\":\"us-gaap:TagName\"}.\n\n"
            f"Entity: {entity}\n"
            f"Entity type: {entity_type}\n"
            f"Evidence: {evidence[:1800]}\n\n"
            f"Candidate tags:\n{candidate_lines}\n"
        )

    @staticmethod
    def _parse_tag(text: str) -> str | None:
        match = re.search(r"us-gaap:[A-Za-z0-9_]+", text)
        return match.group(0) if match else None
