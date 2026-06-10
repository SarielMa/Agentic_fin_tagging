from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .agents import TagSelectorAgent, ValidatorCorrectorAgent
from .data import load_taxonomy
from .evaluation import evaluate_agentic_records, write_agentic_breakdown
from .evidence import build_evidence_builder
from .memory_store import LTMStore
from .retrieval import DynamicLTMRetriever


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    taxonomy_jsonl: Path
    output_dir: Path
    memory_build_csv: Path | None = None
    test_csv: Path | None = None
    stream_csv: Path | None = None
    selector_backend: str = "retrieval"
    selector_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    validator_backend: str = "rule"
    validator_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    table_evidence_backend: str = "heuristic"
    table_evidence_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    top_k: int = 200
    rerank_k: int = 200
    memory_k: int = 8
    bm25_weight: float = 1.0
    dense_weight: float = 0.0
    dense_model: str = ""
    taxonomy_doc_mode: str = "full"
    memory_weight: float = 0.10
    error_weight: float = 0.05
    table_pattern_weight: float = 0.05
    max_iters: int = 2
    supervised_memory_iters: int = 0
    save_top_k: int = 200
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    limit: int = 0
    resume_ltm: bool = False


class AgenticExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        ltm_dir = config.output_dir / "ltm"
        if not config.resume_ltm and self._ltm_has_records(ltm_dir):
            raise ValueError(
                f"Output directory already contains LTM records: {ltm_dir}. "
                "Use a fresh --output-dir, remove the old output, or pass --resume-ltm."
            )
        self.ltm = LTMStore(ltm_dir, write_enabled=True)
        self.taxonomy = load_taxonomy(config.taxonomy_jsonl)
        self.retriever = DynamicLTMRetriever(
            self.taxonomy,
            self.ltm,
            bm25_weight=config.bm25_weight,
            dense_weight=config.dense_weight,
            dense_model=config.dense_model,
            taxonomy_doc_mode=config.taxonomy_doc_mode,
        )
        self.evidence_builder = build_evidence_builder(config.table_evidence_backend, config.table_evidence_model)
        self.selector = TagSelectorAgent(config.selector_backend, config.selector_model)
        self.validator = ValidatorCorrectorAgent(config.validator_backend, config.validator_model)

    def run(self) -> dict[str, Any]:
        if self.config.mode == "offline":
            return self._run_offline()
        if self.config.mode == "online_with_gt":
            return self._run_online_with_gt()
        if self.config.mode == "online_without_gt":
            return self._run_online_without_gt()
        raise ValueError(f"Unsupported mode: {self.config.mode}")

    def _run_offline(self) -> dict[str, Any]:
        if self.config.memory_build_csv is None or self.config.test_csv is None:
            raise ValueError("Offline mode requires --memory-build and --test.")

        self.ltm.set_write_enabled(True)
        build_metrics = self._run_dataset(
            csv_path=self.config.memory_build_csv,
            phase="memory_build",
            agent_mode="offline_build",
            supervise_after_loop=True,
            score=True,
        )
        build_snapshot = self.ltm.snapshot()

        self.ltm.set_write_enabled(False)
        test_metrics = self._run_dataset(
            csv_path=self.config.test_csv,
            phase="test",
            agent_mode="offline_test",
            supervise_after_loop=False,
            score=True,
        )

        summary = {
            "mode": "offline",
            "memory_after_build": build_snapshot,
            "memory_build": build_metrics,
            "test": test_metrics,
            "ltm_dir": str(self.config.output_dir / "ltm"),
        }
        self._write_json("summary.json", summary)
        return summary

    def _run_online_with_gt(self) -> dict[str, Any]:
        if self.config.stream_csv is None:
            raise ValueError("Online modes require --stream.")
        self.ltm.set_write_enabled(True)
        metrics = self._run_dataset(
            csv_path=self.config.stream_csv,
            phase="online_with_gt",
            agent_mode="online_with_gt",
            supervise_after_loop=True,
            score=True,
        )
        summary = {
            "mode": "online_with_gt",
            "stream": metrics,
            "memory_after_stream": self.ltm.snapshot(),
            "ltm_dir": str(self.config.output_dir / "ltm"),
        }
        self._write_json("summary.json", summary)
        return summary

    def _run_online_without_gt(self) -> dict[str, Any]:
        if self.config.stream_csv is None:
            raise ValueError("Online modes require --stream.")
        self.ltm.set_write_enabled(True)
        metrics = self._run_dataset(
            csv_path=self.config.stream_csv,
            phase="online_without_gt",
            agent_mode="online_without_gt",
            supervise_after_loop=False,
            score="answer" in pd.read_csv(self.config.stream_csv, nrows=0).columns,
        )
        summary = {
            "mode": "online_without_gt",
            "stream": metrics,
            "memory_after_stream": self.ltm.snapshot(),
            "ltm_dir": str(self.config.output_dir / "ltm"),
        }
        self._write_json("summary.json", summary)
        return summary

    def _run_dataset(
        self,
        csv_path: Path,
        phase: str,
        agent_mode: str,
        supervise_after_loop: bool,
        score: bool,
    ) -> dict[str, Any]:
        df = pd.read_csv(csv_path)
        phase_dir = self.config.output_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = phase_dir / "predictions.jsonl"

        records: list[dict[str, Any]] = []
        row_limit = min(len(df), self.config.limit) if self.config.limit else len(df)
        progress = tqdm(
            enumerate(df.itertuples(index=False), start=1),
            total=row_limit,
            desc=f"{self.config.mode}:{phase}",
            unit="row",
            dynamic_ncols=True,
        )
        with predictions_path.open("w", encoding="utf-8") as f:
            for row_idx, row in progress:
                record = self._run_one(row_idx, row, agent_mode, supervise_after_loop)
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.set_postfix(
                    correct=record["correct"],
                    prediction=record["prediction"]["Tag"][:40],
                    refresh=False,
                )
                if self.config.limit and row_idx >= self.config.limit:
                    break

        metrics = evaluate_agentic_records(
            records,
            score,
            self.config.recall_k,
            metadata={
                "selector_backend": self.config.selector_backend,
                "selector_model": self.config.selector_model if self.config.selector_backend == "llama" else None,
                "validator_backend": self.config.validator_backend,
                "validator_model": self.config.validator_model if self.config.validator_backend == "llama" else None,
                "table_evidence_backend": self.config.table_evidence_backend,
                "table_evidence_model": self.config.table_evidence_model
                if self.config.table_evidence_backend == "llama"
                else None,
                "top_k": self.config.top_k,
                "rerank_k": self.config.rerank_k,
                "retrieval": "bm25" if self.config.dense_weight <= 0 else "hybrid_bm25_dense",
                "bm25_weight": self.config.bm25_weight,
                "dense_weight": self.config.dense_weight,
                "dense_model": (self.config.dense_model or "svd_fallback") if self.config.dense_weight > 0 else None,
                "taxonomy_doc_mode": self.config.taxonomy_doc_mode,
                "max_iters": self.config.max_iters,
                "supervised_memory_iters": self.config.supervised_memory_iters,
                "resume_ltm": self.config.resume_ltm,
            },
        )
        metrics["predictions_path"] = str(predictions_path)
        metrics["memory_snapshot"] = self.ltm.snapshot()
        with (phase_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_agentic_breakdown(records, phase_dir, score)
        return metrics

    def _run_one(self, row_idx: int, row: Any, agent_mode: str, supervise_after_loop: bool) -> dict[str, Any]:
        pre_evidence_table_patterns = self.retriever.retrieve_table_patterns_for_evidence(
            row.context,
            row.category,
            row.entity,
            row.entity_type,
            top_k=self.config.memory_k,
        )
        evidence = self.evidence_builder.build(
            row.context,
            row.category,
            row.entity,
            row.entity_type,
            table_patterns=pre_evidence_table_patterns,
        )
        raw_candidates = self.retriever.retrieve_taxonomy(row.entity_type, evidence, top_k=self.config.top_k)
        feedback: list[str] = []
        attempts = []
        final_decision = None
        candidates: list[dict[str, Any]] = []
        memory_hits: dict[str, list[dict[str, Any]]] = {}

        for attempt in range(1, self.config.max_iters + 1):
            candidates, memory_hits = self.retriever.retrieve(
                row.entity,
                row.entity_type,
                evidence,
                top_k=self.config.top_k,
                memory_k=self.config.memory_k,
                memory_weight=self.config.memory_weight,
                error_weight=self.config.error_weight,
                table_pattern_weight=self.config.table_pattern_weight,
            )
            selection = self.selector.select(
                row.entity,
                row.entity_type,
                evidence,
                candidates[: self.config.rerank_k],
                feedback,
                memory_hits=memory_hits,
            )
            decision = self.validator.validate(
                mode=agent_mode,
                category=row.category,
                entity=row.entity,
                entity_type=row.entity_type,
                evidence=evidence,
                selection=selection,
                candidates=candidates,
                memory_hits=memory_hits,
                gold_tag=None,
                attempt=attempt,
                max_iters=self.config.max_iters,
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "selection": selection.__dict__,
                    "decision": {
                        "action": decision.action,
                        "final_tag": decision.final_tag,
                        "passed": decision.passed,
                        "rationale": decision.rationale,
                        "feedback_to_selector": decision.feedback_to_selector,
                        "flags": decision.flags,
                        "memory_writes": [write.namespace for write in decision.memory_writes],
                    },
                }
            )
            if decision.memory_writes:
                self.ltm.append_many(decision.memory_writes)
            final_decision = decision
            if decision.action in {"keep", "correct", "flag"}:
                break
            feedback.append(decision.feedback_to_selector)

        final_tag = final_decision.final_tag if final_decision is not None else ""
        gold_tag = row.answer if hasattr(row, "answer") else None
        supervision_record = None
        if supervise_after_loop and gold_tag is not None:
            refinement_record = self._run_supervised_memory_refinement(
                row=row,
                evidence=evidence,
                original_final_tag=final_tag,
                gold_tag=gold_tag,
                candidates=candidates,
                memory_hits=memory_hits,
            )
            supervision = self.validator.supervise_after_loop(
                mode=agent_mode,
                category=row.category,
                entity=row.entity,
                entity_type=row.entity_type,
                evidence=evidence,
                predicted_tag=final_tag,
                gold_tag=gold_tag,
                flags=final_decision.flags if final_decision else [],
                memory_lesson=refinement_record["lesson"] if refinement_record else "",
            )
            if supervision.memory_writes:
                self.ltm.append_many(supervision.memory_writes)
            supervision_record = {
                "action": supervision.action,
                "final_tag": supervision.final_tag,
                "passed": supervision.passed,
                "rationale": supervision.rationale,
                "flags": supervision.flags,
                "memory_writes": [write.namespace for write in supervision.memory_writes],
                "supervised_refinement": refinement_record,
            }
        return {
            "row_index": row_idx - 1,
            "gold": {"Fact": str(row.entity), "Type": row.entity_type, "Tag": gold_tag},
            "prediction": {"Fact": str(row.entity), "Type": row.entity_type, "Tag": final_tag},
            "correct": final_tag == gold_tag if gold_tag is not None else None,
            "stm": {
                "context_id": f"{agent_mode}-{row_idx}",
                "category": row.category,
                "entity": {"value": str(row.entity), "type": row.entity_type},
                "evidence": evidence,
                "pre_evidence_table_pattern_hits": pre_evidence_table_patterns[:5],
                "raw_top_k": raw_candidates[: self.config.save_top_k],
                "top_k": candidates[: self.config.save_top_k],
                "memory_hits": {key: value[:5] for key, value in memory_hits.items()},
                "attempts": attempts,
                "final_action": final_decision.action if final_decision else "none",
                "final_flags": final_decision.flags if final_decision else [],
                "post_loop_supervision": supervision_record,
            },
        }

    def _run_supervised_memory_refinement(
        self,
        row: Any,
        evidence: str,
        original_final_tag: str,
        gold_tag: str,
        candidates: list[dict[str, Any]],
        memory_hits: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        if self.config.supervised_memory_iters <= 0:
            return None

        supervised_candidates, gold_candidate_action = self._supervised_candidates(candidates, gold_tag)
        if not supervised_candidates:
            return None

        feedback = [
            self._teacher_feedback(
                selected_tag=original_final_tag,
                gold_tag=gold_tag,
                supervised_candidates=supervised_candidates,
                first=True,
            )
        ]
        attempts: list[dict[str, Any]] = []
        reached_gold = False

        for refine_iter in range(1, self.config.supervised_memory_iters + 1):
            selection = self.selector.select(
                row.entity,
                row.entity_type,
                evidence,
                supervised_candidates[: self.config.rerank_k],
                feedback,
                memory_hits=memory_hits,
            )
            selected_tag = selection.selected_tag
            is_gold = selected_tag == gold_tag
            attempts.append(
                {
                    "iteration": refine_iter,
                    "selected_tag": selected_tag,
                    "reached_gold": is_gold,
                    "rationale": selection.rationale,
                }
            )
            if is_gold:
                reached_gold = True
                break
            feedback.append(
                self._teacher_feedback(
                    selected_tag=selected_tag,
                    gold_tag=gold_tag,
                    supervised_candidates=supervised_candidates,
                    first=False,
                )
            )

        lesson = self._supervised_memory_lesson(
            original_final_tag=original_final_tag,
            gold_tag=gold_tag,
            evidence=evidence,
            supervised_candidates=supervised_candidates,
            gold_candidate_action=gold_candidate_action,
            reached_gold=reached_gold,
        )
        return {
            "iterations_requested": self.config.supervised_memory_iters,
            "iterations_run": len(attempts),
            "gold_candidate_action": gold_candidate_action,
            "selector_reached_gold": reached_gold,
            "attempts": attempts,
            "lesson": lesson,
        }

    def _supervised_candidates(
        self,
        candidates: list[dict[str, Any]],
        gold_tag: str,
    ) -> tuple[list[dict[str, Any]], str]:
        supervised = [dict(candidate) for candidate in candidates]
        if not supervised:
            supervised = []

        top_score = max((float(candidate.get("score", 0.0)) for candidate in supervised), default=0.0)
        gold_candidate = next((candidate for candidate in supervised if candidate.get("tag") == gold_tag), None)
        if gold_candidate is not None:
            gold_candidate["score"] = top_score + 1.0
            action = "boosted_existing_gold_candidate"
        else:
            concept = self.retriever.tag_to_taxonomy_idx.get(gold_tag)
            if concept is None:
                return supervised, "gold_tag_not_in_taxonomy"
            taxonomy_concept = self.taxonomy[concept]
            supervised.append(
                {
                    "rank": 0,
                    "tag": taxonomy_concept.tag,
                    "entity_type": taxonomy_concept.entity_type,
                    "text": taxonomy_concept.text,
                    "score": top_score + 1.0,
                    "supervised_gold_injected": True,
                }
            )
            action = "injected_gold_candidate"

        supervised.sort(key=lambda candidate: float(candidate.get("score", 0.0)), reverse=True)
        for rank, candidate in enumerate(supervised, start=1):
            candidate["rank"] = rank
        return supervised, action

    @staticmethod
    def _teacher_feedback(
        selected_tag: str,
        gold_tag: str,
        supervised_candidates: list[dict[str, Any]],
        first: bool,
    ) -> str:
        gold_text = AgenticExperiment._candidate_text(supervised_candidates, gold_tag)
        if selected_tag == gold_tag:
            return (
                "Supervised memory feedback: the selected tag matches the gold tag. "
                f"Use this as a positive example for {gold_tag}. Gold concept text: {gold_text}"
            )
        prefix = "Supervised memory feedback:" if first else "Still supervised feedback:"
        selected_text = AgenticExperiment._candidate_text(supervised_candidates, selected_tag)
        return (
            f"{prefix} the previous selected tag {selected_tag} was wrong. "
            f"The correct gold tag is {gold_tag}. "
            f"Prefer the gold concept text: {gold_text}. "
            f"Avoid the wrong concept text: {selected_text}."
        )

    @staticmethod
    def _supervised_memory_lesson(
        original_final_tag: str,
        gold_tag: str,
        evidence: str,
        supervised_candidates: list[dict[str, Any]],
        gold_candidate_action: str,
        reached_gold: bool,
    ) -> str:
        gold_text = AgenticExperiment._candidate_text(supervised_candidates, gold_tag)
        evidence_text = " ".join(str(evidence).split())[:700]
        parts = [
            f"Supervised memory lesson: correct tag is {gold_tag}.",
            f"Gold concept text: {gold_text}.",
        ]
        if original_final_tag != gold_tag:
            wrong_text = AgenticExperiment._candidate_text(supervised_candidates, original_final_tag)
            parts.append(f"Avoid prior prediction {original_final_tag}. Wrong concept text: {wrong_text}.")
        else:
            parts.append("The original prediction matched the gold tag; store this as a positive example.")
        if gold_candidate_action == "injected_gold_candidate":
            parts.append("The gold tag was injected for supervised memory refinement because retrieval did not surface it.")
        elif gold_candidate_action == "boosted_existing_gold_candidate":
            parts.append("The gold tag was already retrieved and was boosted for supervised memory refinement.")
        parts.append(f"Selector reached gold during refinement: {reached_gold}.")
        parts.append(f"Evidence cues: {evidence_text}")
        return " ".join(parts)

    @staticmethod
    def _candidate_text(candidates: list[dict[str, Any]], tag: str) -> str:
        candidate = next((item for item in candidates if item.get("tag") == tag), None)
        if candidate is None:
            return ""
        return str(candidate.get("text", ""))

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        with (self.config.output_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _ltm_has_records(ltm_dir: Path) -> bool:
        return any(
            (ltm_dir / f"{namespace}.jsonl").exists()
            and (ltm_dir / f"{namespace}.jsonl").stat().st_size > 0
            for namespace in LTMStore.VALID_NAMESPACES
        )
