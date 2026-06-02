from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import TagSelectorAgent, ValidatorCorrectorAgent
from .data import load_taxonomy
from .memory_store import LTMStore
from .retrieval import DynamicLTMRetriever
from .text_utils import localize_context


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
    top_k: int = 200
    rerank_k: int = 20
    memory_k: int = 8
    memory_weight: float = 0.10
    error_weight: float = 0.05
    table_pattern_weight: float = 0.05
    max_iters: int = 2
    save_top_k: int = 200
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    limit: int = 0


class AgenticExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.ltm = LTMStore(config.output_dir / "ltm", write_enabled=True)
        self.taxonomy = load_taxonomy(config.taxonomy_jsonl)
        self.retriever = DynamicLTMRetriever(self.taxonomy, self.ltm)
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
            gold_visible=True,
            score=True,
        )
        build_snapshot = self.ltm.snapshot()

        self.ltm.set_write_enabled(False)
        test_metrics = self._run_dataset(
            csv_path=self.config.test_csv,
            phase="test",
            agent_mode="offline_test",
            gold_visible=False,
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
            gold_visible=True,
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
            gold_visible=False,
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
        gold_visible: bool,
        score: bool,
    ) -> dict[str, Any]:
        df = pd.read_csv(csv_path)
        phase_dir = self.config.output_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = phase_dir / "predictions.jsonl"

        records: list[dict[str, Any]] = []
        with predictions_path.open("w", encoding="utf-8") as f:
            for row_idx, row in enumerate(df.itertuples(index=False), start=1):
                record = self._run_one(row_idx, row, agent_mode, gold_visible)
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if self.config.limit and row_idx >= self.config.limit:
                    break

        metrics = self._metrics(records, score)
        metrics["predictions_path"] = str(predictions_path)
        metrics["memory_snapshot"] = self.ltm.snapshot()
        with (phase_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        self._write_breakdown(records, phase_dir, score)
        return metrics

    def _run_one(self, row_idx: int, row: Any, agent_mode: str, gold_visible: bool) -> dict[str, Any]:
        evidence = localize_context(row.context, row.category, row.entity)
        feedback: list[str] = []
        attempts = []
        final_decision = None
        candidates: list[dict[str, Any]] = []
        memory_hits: dict[str, list[dict[str, Any]]] = {}
        gold_for_agent = row.answer if gold_visible and hasattr(row, "answer") else None

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
                gold_tag=gold_for_agent,
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
                "top_k": candidates[: self.config.save_top_k],
                "memory_hits": {key: value[:5] for key, value in memory_hits.items()},
                "attempts": attempts,
                "final_action": final_decision.action if final_decision else "none",
                "final_flags": final_decision.flags if final_decision else [],
            },
        }

    def _metrics(self, records: list[dict[str, Any]], score: bool) -> dict[str, Any]:
        n = len(records)
        metrics: dict[str, Any] = {
            "num_examples": n,
            "selector_backend": self.config.selector_backend,
            "selector_model": self.config.selector_model if self.config.selector_backend == "llama" else None,
            "validator_backend": self.config.validator_backend,
            "validator_model": self.config.validator_model if self.config.validator_backend == "llama" else None,
            "top_k": self.config.top_k,
            "rerank_k": self.config.rerank_k,
            "max_iters": self.config.max_iters,
        }
        action_counts: dict[str, int] = {}
        for record in records:
            action = record["stm"]["final_action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        metrics["action_counts"] = action_counts
        metrics["flag_rate"] = action_counts.get("flag", 0) / n if n else math.nan

        if not score:
            return metrics

        correct = sum(1 for record in records if record["correct"])
        recall_counts = {k: 0 for k in self.config.recall_k}
        for record in records:
            gold_tag = record["gold"]["Tag"]
            candidate_tags = [candidate["tag"] for candidate in record["stm"]["top_k"]]
            for k in self.config.recall_k:
                recall_counts[k] += int(gold_tag in candidate_tags[:k])
        metrics["tag_accuracy"] = correct / n if n else math.nan
        metrics["recall_at_k"] = {str(k): recall_counts[k] / n if n else math.nan for k in self.config.recall_k}
        return metrics

    def _write_breakdown(self, records: list[dict[str, Any]], phase_dir: Path, score: bool) -> None:
        if not records or not score:
            return
        rows = []
        for record in records:
            candidates = [candidate["tag"] for candidate in record["stm"]["top_k"]]
            rows.append(
                {
                    "category": record["stm"]["category"],
                    "entity_type": record["gold"]["Type"],
                    "correct": record["correct"],
                    "flagged": record["stm"]["final_action"] == "flag",
                    "recall20": record["gold"]["Tag"] in candidates[:20],
                    "recall50": record["gold"]["Tag"] in candidates[:50],
                    "recall100": record["gold"]["Tag"] in candidates[:100],
                    "recall200": record["gold"]["Tag"] in candidates[:200],
                }
            )
        df = pd.DataFrame(rows)
        breakdown = {
            "by_category": self._group_breakdown(df, "category"),
            "by_entity_type": self._group_breakdown(df, "entity_type"),
        }
        with (phase_dir / "breakdown.json").open("w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2)

    @staticmethod
    def _group_breakdown(df: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for key, group in df.groupby(column):
            result[str(key)] = {
                "n": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "recall20": float(group["recall20"].mean()),
                "recall50": float(group["recall50"].mean()),
                "recall100": float(group["recall100"].mean()),
                "recall200": float(group["recall200"].mean()),
                "flag_rate": float(group["flagged"].mean()),
            }
        return result

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        with (self.config.output_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
