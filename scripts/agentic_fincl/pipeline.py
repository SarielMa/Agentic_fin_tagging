# Orchestrates offline, online_gt, and online_wo_gt three-agent runs.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .agents import RetrieverAgent, SelectorAgent, ValidatorAgent
from .data import Example, load_examples, load_taxonomy
from .evaluation import evaluate_records
from .memory_store import LTMStore
from .text_utils import context_text


@dataclass(frozen=True)
class BaselineConfig:
    mode: str
    taxonomy_jsonl: Path
    output_dir: Path
    memory_build_csv: Path
    test_csv: Path
    stream_csv: Path | None
    selector_model: str
    validator_model: str
    top_k: int = 200
    memory_k: int = 5
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    limit: int = 0
    coach_mode: str = "hint"
    coach_retries: int = 2
    oracle_fallback: bool = True


class BaselinePipeline:
    """Overall pipeline for the minimal three-agent baseline."""

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        taxonomy = load_taxonomy(config.taxonomy_jsonl)
        self.retriever = RetrieverAgent(taxonomy)
        self.selector = SelectorAgent(config.selector_model, memory_k=config.memory_k)
        self.validator = ValidatorAgent(config.validator_model, memory_k=config.memory_k)
        self.ltm = LTMStore(config.output_dir / "ltm")

        if self.ltm.has_records():
            raise RuntimeError(
                f"{config.output_dir / 'ltm'} already contains memory records. "
                "Use a new output directory or remove the old run."
            )

    def run(self) -> dict[str, Any]:
        if self.config.mode == "offline":
            summary = {
                "mode": "offline",
                "memory_build": self._run_memory_build(
                    load_examples(self.config.memory_build_csv, self.config.limit),
                    self.config.output_dir / "memory_build",
                ),
                "test": self._run_testing(
                    load_examples(self.config.test_csv, self.config.limit),
                    self.config.output_dir / "test",
                ),
            }
        elif self.config.mode == "online_gt":
            stream_path = self.config.stream_csv or self.config.test_csv
            summary = {
                "mode": "online_gt",
                "online_gt": self._run_memory_build(
                    load_examples(stream_path, self.config.limit),
                    self.config.output_dir / "online_gt",
                ),
            }
        elif self.config.mode == "online_wo_gt":
            stream_path = self.config.stream_csv or self.config.test_csv
            summary = {
                "mode": "online_wo_gt",
                "online_wo_gt": self._run_testing(
                    load_examples(stream_path, self.config.limit),
                    self.config.output_dir / "online_wo_gt",
                ),
            }
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        summary["ltm_counts"] = self.ltm.snapshot()
        summary_path = self.config.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def _run_memory_build(self, examples: list[Example], phase_dir: Path) -> dict[str, Any]:
        """Supervised phase: one selector pass, then validator writes one memory."""

        records = []
        for example in tqdm(examples, desc=f"{self.config.mode} memory-build", unit="example"):
            record = self._memory_build_one(example)
            records.append(record)
        return self._write_phase(
            phase_dir=phase_dir,
            records=records,
            phase="memory_build",
            iterations=2,
        )

    def _run_testing(self, examples: list[Example], phase_dir: Path) -> dict[str, Any]:
        """Unsupervised phase: selector, validator feedback, selector retry."""

        records = []
        for example in tqdm(examples, desc=f"{self.config.mode} testing", unit="example"):
            record = self._testing_one(example)
            records.append(record)
        return self._write_phase(
            phase_dir=phase_dir,
            records=records,
            phase="testing",
            iterations=2,
        )

    def _memory_build_one(self, example: Example) -> dict[str, Any]:
        """Supervised teach-then-retry: while the student is wrong, the GT-aware critic
        coaches up to ``coach_retries`` retries, each hint told which tags were already
        rejected so it escalates. A successful retry lands the example in correct_book.

        If the student still fails and ``oracle_fallback`` is set, the ground-truth
        (label -> tag) exemplar is banked in correct_book anyway (its value is the
        mapping, not whether the student earned it) and the contrastive miss is written
        to error_book. This de-starves the positive book without leaking GT at test time."""

        context = context_text(example.context)
        candidates = self.retriever.retrieve(context, example.entity_type, self.config.top_k)

        first = self.selector.select(example, context, candidates, self.ltm)
        attempts = [first]
        coachings: list[Any] = []
        final_tag = first.tag

        has_gt = example.answer is not None
        tried = [first.tag]
        while has_gt and final_tag != example.answer and len(coachings) < self.config.coach_retries:
            coaching = self.validator.coach_with_gt(
                example=example,
                context=context,
                candidates=candidates,
                selector_tag=final_tag,
                tried_tags=tried,
                coach_mode=self.config.coach_mode,
            )
            coachings.append(coaching)
            retry = self.selector.select(
                example=example,
                context=context,
                candidates=candidates,
                ltm=self.ltm,
                feedback=coaching.hint,
            )
            attempts.append(retry)
            final_tag = retry.tag
            tried.append(final_tag)

        earned = has_gt and final_tag == example.answer
        if earned or not has_gt:
            write_result = self.validator.write_with_gt(example, context, final_tag, self.ltm)
        elif self.config.oracle_fallback:
            write_result = self.validator.write_fallback_with_gt(example, context, final_tag, self.ltm)
        else:
            write_result = self.validator.write_with_gt(example, context, final_tag, self.ltm)

        return self._record(
            example=example,
            context=context,
            candidates=candidates,
            final_tag=final_tag,
            selector_attempts=attempts,
            validator_feedback=coachings[-1] if coachings else None,
            memory_write=write_result,
            phase="memory_build",
        )

    def _testing_one(self, example: Example) -> dict[str, Any]:
        context = context_text(example.context)
        candidates = self.retriever.retrieve(context, example.entity_type, self.config.top_k)

        first_selection = self.selector.select(example, context, candidates, self.ltm)
        feedback = self.validator.review_without_gt(
            example=example,
            context=context,
            candidates=candidates,
            selector_tag=first_selection.tag,
            ltm=self.ltm,
        )
        second_selection = self.selector.select(
            example=example,
            context=context,
            candidates=candidates,
            ltm=self.ltm,
            feedback=f"Validator suggested {feedback.suggested_tag}. {feedback.feedback}",
        )

        return self._record(
            example=example,
            context=context,
            candidates=candidates,
            final_tag=second_selection.tag,
            selector_attempts=[first_selection, second_selection],
            validator_feedback=feedback,
            memory_write=None,
            phase="testing",
        )

    def _record(
        self,
        example: Example,
        context: str,
        candidates: list[dict[str, Any]],
        final_tag: str,
        selector_attempts: list[Any],
        validator_feedback: Any | None,
        memory_write: Any | None,
        phase: str,
    ) -> dict[str, Any]:
        gold_tag = example.answer
        return {
            "row_index": example.row_index,
            "phase": phase,
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
                "Tag": final_tag,
            },
            "correct": bool(gold_tag and final_tag == gold_tag),
            "retrieval": {
                "method": "type_filter_then_bm25_context_to_taxonomy_text",
                "top_k": candidates,
            },
            "selector": {
                "attempts": [asdict(selection) for selection in selector_attempts],
            },
            "validator": asdict(validator_feedback) if validator_feedback is not None else None,
            "memory_write": asdict(memory_write) if memory_write is not None else None,
            "ltm_counts": self.ltm.snapshot(),
        }

    def _write_phase(
        self,
        phase_dir: Path,
        records: list[dict[str, Any]],
        phase: str,
        iterations: int,
    ) -> dict[str, Any]:
        phase_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = phase_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = evaluate_records(
            records,
            self.config.recall_k,
            metadata={
                "phase": phase,
                "iterations": iterations,
                "retrieval": "bm25",
                "retrieval_agent": "type_filter_then_bm25_context_to_taxonomy_text",
                "top_k": self.config.top_k,
                "memory_k": self.config.memory_k,
                "selector_model": self.config.selector_model,
                "validator_model": self.config.validator_model,
                "predictions_path": str(predictions_path),
                "ltm_counts": self.ltm.snapshot(),
            },
        )
        with (phase_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics
