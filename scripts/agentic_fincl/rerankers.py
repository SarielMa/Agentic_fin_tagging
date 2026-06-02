from __future__ import annotations

import re
from typing import Any, Protocol


class Reranker(Protocol):
    def choose(self, entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        ...


class RetrievalTop1Reranker:
    """No-LLM baseline: choose the highest-scoring retrieved candidate."""

    def choose(self, entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        return candidates[0]["tag"] if candidates else ""


class LlamaReranker:
    """Local Hugging Face Llama reranker.

    This is intended for the GPU session. It uses local_files_only=True so the
    run is reproducible and does not try to download model weights.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 48,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() and device is None else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.to(self.device)
        else:
            self.device = None
        self.model.eval()

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
        if self.device is not None:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = self._parse_tag(generated)
        return parsed or candidates[0]["tag"]

    @staticmethod
    def _build_prompt(entity: Any, entity_type: str, evidence: str, candidates: list[dict[str, Any]]) -> str:
        candidate_lines = "\n".join(
            f"{i}. {candidate['tag']} | {candidate['text']}" for i, candidate in enumerate(candidates, start=1)
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
