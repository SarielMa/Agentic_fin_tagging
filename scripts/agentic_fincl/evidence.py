from __future__ import annotations

import copy
import json
import re
from typing import Any, Protocol

from .hf_generation import load_local_causal_lm, model_input_device, move_inputs_to_device
from .text_utils import (
    localize_context,
    normalize_space,
    rewrite_evidence_for_retrieval,
    row_contains_entity,
    table_rows,
)


class EvidenceBuilder(Protocol):
    def build(
        self,
        context: str,
        category: str,
        entity: Any,
        entity_type: str,
        table_patterns: list[dict[str, Any]] | None = None,
    ) -> str:
        ...


class HeuristicEvidenceBuilder:
    """Fast default evidence builder."""

    def build(
        self,
        context: str,
        category: str,
        entity: Any,
        entity_type: str,
        table_patterns: list[dict[str, Any]] | None = None,
    ) -> str:
        evidence = localize_context(context, category, entity)
        return rewrite_evidence_for_retrieval(evidence, category, entity, entity_type)


class LlamaTableEvidenceBuilder:
    """LLM evidence builder for table rows, with heuristic fallback.

    The LLM sees context, category, entity, entity_type, and optional table
    pattern memory from previous samples. It never sees the current sample's
    gold US-GAAP answer.
    """

    def __init__(
        self,
        model_name: str,
        fallback: EvidenceBuilder | None = None,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.fallback = fallback or HeuristicEvidenceBuilder()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer, self.model, self.device = load_local_causal_lm(
            model_name,
            torch,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

    def build(
        self,
        context: str,
        category: str,
        entity: Any,
        entity_type: str,
        table_patterns: list[dict[str, Any]] | None = None,
    ) -> str:
        if category != "table":
            return self.fallback.build(context, category, entity, entity_type)

        table_view = self._compact_table_view(context, entity)
        prompt = self._build_prompt(table_view, entity, entity_type, table_patterns or [])
        raw = self._generate(prompt)
        parsed = self._parse_json(raw)
        if not parsed:
            return self.fallback.build(context, category, entity, entity_type)
        return self._format_evidence(parsed, entity, entity_type)

    def _generate(self, prompt: str) -> str:
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
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    @staticmethod
    def _compact_table_view(context: str, entity: Any) -> str:
        rows = table_rows(context)
        if not rows:
            return localize_context(context, "table", entity, max_chars=3500)

        matched = [idx for idx, row in enumerate(rows) if row_contains_entity(row, entity)]
        selected: dict[int, str] = {}
        for idx, row in enumerate(rows[:8]):
            selected[idx] = row
        for idx in matched:
            start = max(0, idx - 5)
            end = min(len(rows), idx + 4)
            for row_idx in range(start, end):
                selected[row_idx] = rows[row_idx]

        lines = [f"row {idx}: {normalize_space(row)}" for idx, row in sorted(selected.items())]
        if matched:
            lines.append("matched row indices: " + ", ".join(str(idx) for idx in matched))
        return "\n".join(lines)[:7000]

    @staticmethod
    def _format_table_patterns(table_patterns: list[dict[str, Any]]) -> str:
        lines = []
        for idx, pattern in enumerate(table_patterns[:3], start=1):
            score = pattern.get("score")
            try:
                score_text = f" score={float(score):.3f}" if score is not None else ""
            except (TypeError, ValueError):
                score_text = ""
            fields = [
                f"memory {idx}{score_text}:",
                f"prior_entity={pattern.get('entity', '')}",
                f"prior_tag={pattern.get('tag', '')}",
                f"prior_pattern={str(pattern.get('pattern', ''))[:500]}",
                f"prior_evidence={str(pattern.get('evidence', ''))[:700]}",
            ]
            lines.append(normalize_space(" ".join(field for field in fields if field)))
        return "\n".join(lines)

    @classmethod
    def _build_prompt(
        cls,
        table_view: str,
        entity: Any,
        entity_type: str,
        table_patterns: list[dict[str, Any]],
    ) -> str:
        memory_text = cls._format_table_patterns(table_patterns)
        memory_section = ""
        if memory_text:
            memory_section = (
                "Relevant table-pattern memory from previous samples:\n"
                f"{memory_text}\n\n"
                "Use these memories as examples of which title, unit, headers, section, row label, "
                "and nearby rows were useful for similar tables. Do not output a US-GAAP tag.\n\n"
            )
        return (
            "You are building evidence for US-GAAP XBRL tagging. "
            "Do not predict the US-GAAP tag. Extract table context only.\n\n"
            f"Target entity: {entity}\n"
            f"Target entity type: {entity_type}\n\n"
            f"{memory_section}"
            "The table below is serialized as numbered rows. Identify the target value's useful context.\n"
            "Return only valid JSON with these keys:\n"
            "{\n"
            '  "table_title": "...",\n'
            '  "unit": "...",\n'
            '  "column_header": "...",\n'
            '  "section_context": "...",\n'
            '  "matched_row": "...",\n'
            '  "nearby_rows": "...",\n'
            '  "retrieval_query": "..."\n'
            "}\n\n"
            "The retrieval_query should be a compact natural-language description useful for matching the "
            "target entity to a US-GAAP taxonomy concept. Include table title, unit, column/year, section, "
            "and row label when available.\n\n"
            f"Table rows:\n{table_view}\n"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _format_evidence(parsed: dict[str, Any], entity: Any, entity_type: str) -> str:
        row_text = normalize_space(
            " ".join(
                str(parsed.get(key, ""))
                for key in ("table_title", "section_context", "matched_row", "nearby_rows", "retrieval_query")
            )
        )
        fields = [
            ("format", "table_as_text"),
            ("entity", str(entity)),
            ("entity_type", entity_type),
            ("target_value", str(entity)),
            ("table_title", parsed.get("table_title", "")),
            ("unit", parsed.get("unit", "")),
            ("column_header", parsed.get("column_header", "")),
            ("section_context", parsed.get("section_context", "")),
            ("matched_row", parsed.get("matched_row", "")),
            ("nearby_rows", parsed.get("nearby_rows", "")),
            ("retrieval_query", parsed.get("retrieval_query", "")),
            ("text_evidence", row_text),
        ]
        return normalize_space(" ".join(f"{key}: {value}" for key, value in fields if value))


def build_evidence_builder(backend: str, model: str) -> EvidenceBuilder:
    if backend == "heuristic":
        return HeuristicEvidenceBuilder()
    if backend == "llama":
        return LlamaTableEvidenceBuilder(model)
    raise ValueError(f"Unsupported table evidence backend: {backend}")
