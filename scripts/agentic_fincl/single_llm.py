# Runs the memoryless single-LLM test path fed only by Agent-1 retrieval.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .agents import (
    LocalLLM,
    RetrieverAgent,
    _clip,
    _extract_candidate_tag,
    _extract_field,
    _first_candidate_tag,
    _format_candidates,
)
from .data import Example, load_examples, load_taxonomy
from .evaluation import evaluate_records
from .text_utils import context_text


@dataclass(frozen=True)
class SingleLLMConfig:
    taxonomy_jsonl: Path
    test_csv: Path
    output_dir: Path
    model: str
    top_k: int = 200
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    limit: int = 0


class SingleLLMTester:
    """Memoryless one-pass LLM tester fed only by Agent-1 retrieval."""

    def __init__(self, config: SingleLLMConfig) -> None:
        self.config = config
        taxonomy = load_taxonomy(config.taxonomy_jsonl)
        self.retriever = RetrieverAgent(taxonomy)
        self.llm = LocalLLM(config.model)

    def run(self) -> dict[str, Any]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        examples = load_examples(self.config.test_csv, self.config.limit)
        records = [
            self._run_one(example)
            for example in tqdm(examples, desc="single_llm testing", unit="example")
        ]

        predictions_path = self.config.output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = evaluate_records(
            records,
            self.config.recall_k,
            metadata={
                "phase": "single_llm_testing",
                "iterations": 1,
                "retrieval": "bm25",
                "retrieval_agent": "type_filter_then_bm25_context_to_taxonomy_text",
                "top_k": self.config.top_k,
                "model": self.config.model,
                "memory": "none",
                "validator": "none",
                "predictions_path": str(predictions_path),
            },
        )
        with (self.config.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        summary = {
            "mode": "single_llm_testing",
            "metrics": metrics,
        }
        with (self.config.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def _run_one(self, example: Example) -> dict[str, Any]:
        context = context_text(example.context)
        candidates = self.retriever.retrieve(context, example.entity_type, self.config.top_k)
        raw_output = self.llm.generate(
            system=(
                "You are a single-LLM US-GAAP tag selector. You receive only the current "
                "context and Agent-1 retrieval candidates. Do not use memory, validation, "
                "or retries. Select exactly one candidate tag. Return compact JSON with "
                "keys tag and rationale."
            ),
            user=_single_llm_prompt(example, context, candidates),
            max_new_tokens=128,
        )
        selected_tag = _extract_candidate_tag(raw_output, candidates) or _first_candidate_tag(candidates)
        rationale = _extract_field(raw_output, "rationale") or _clip(raw_output, 240)
        gold_tag = example.answer

        return {
            "row_index": example.row_index,
            "phase": "single_llm_testing",
            "input": {
                "category": example.category,
                "entity": str(example.entity),
                "entity_type": example.entity_type,
                "context": context,
            },
            "gold": {
                "Fact": str(example.entity),
                "Type": example.entity_type,
                "Tag": gold_tag,
            },
            "prediction": {
                "Fact": str(example.entity),
                "Type": example.entity_type,
                "Tag": selected_tag,
            },
            "correct": bool(gold_tag and selected_tag == gold_tag),
            "retrieval": {
                "method": "type_filter_then_bm25_context_to_taxonomy_text",
                "top_k": candidates,
            },
            "single_llm": {
                "tag": selected_tag,
                "rationale": rationale,
                "raw_output": raw_output,
                "memory": None,
                "validator": None,
                "iterations": 1,
            },
        }


def _single_llm_prompt(example: Example, context: str, candidates: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"Entity type: {example.entity_type}",
            f"Entity value: {example.entity}",
            "Candidate tags from Agent 1 retrieval:",
            _format_candidates(candidates),
            "Context:",
            _clip(context, 2200),
            'Return JSON only, for example: {"tag": "us-gaap:ExampleTag", "rationale": "..."}',
        ]
    )
