from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .memory_store import MemoryWrite
from .rerankers import LlamaReranker, RetrievalTop1Reranker, Reranker
from .validation import validate_prediction


@dataclass(frozen=True)
class Selection:
    selected_tag: str
    confidence: float
    rationale: str
    backend: str
    memory_context: str = ""


@dataclass(frozen=True)
class ValidationDecision:
    action: str
    final_tag: str
    passed: bool
    rationale: str
    feedback_to_selector: str = ""
    memory_writes: list[MemoryWrite] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


class TagSelectorAgent:
    """Agent 1: read-only tag selector."""

    def __init__(self, backend: str = "retrieval", model: str = "meta-llama/Llama-3.2-3B-Instruct") -> None:
        self.backend = backend
        if backend == "retrieval":
            self.reranker: Reranker = RetrievalTop1Reranker()
        elif backend == "llama":
            self.reranker = LlamaReranker(model)
        else:
            raise ValueError(f"Unsupported selector backend: {backend}")

    def select(
        self,
        entity: Any,
        entity_type: str,
        evidence: str,
        candidates: list[dict[str, Any]],
        feedback: list[str],
        memory_hits: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Selection:
        memory_context = self._selector_memory_context(memory_hits or {})
        evidence_parts = []
        if feedback:
            evidence_parts.append("Validator feedback from prior attempt: " + " ".join(feedback))
        if memory_context:
            evidence_parts.append("Relevant long-term memory lessons for tag selection:\n" + memory_context)
        evidence_parts.append("Current evidence:\n" + evidence)
        augmented_evidence = "\n\n".join(evidence_parts)
        selected_tag = self.reranker.choose(entity, entity_type, augmented_evidence, candidates)
        confidence = self._score_confidence(selected_tag, candidates)
        memory_note = " using retrieved LTM lessons" if memory_context else ""
        return Selection(
            selected_tag=selected_tag,
            confidence=confidence,
            rationale=f"{self.backend} selector chose {selected_tag}{memory_note}",
            backend=self.backend,
            memory_context=memory_context,
        )

    @staticmethod
    def _score_confidence(selected_tag: str, candidates: list[dict[str, Any]]) -> float:
        if not candidates:
            return 0.0
        selected = next((candidate for candidate in candidates if candidate["tag"] == selected_tag), None)
        if selected is None:
            return 0.0
        top_score = candidates[0]["score"]
        if top_score <= 0:
            return 0.0
        return float(max(0.0, min(1.0, selected["score"] / top_score)))

    @classmethod
    def _selector_memory_context(cls, memory_hits: dict[str, list[dict[str, Any]]]) -> str:
        lines: list[str] = []

        selector_hits = memory_hits.get("selector_memory", [])
        if selector_hits:
            lines.append("Prior accepted examples:")
            for hit in selector_hits[:2]:
                lines.append(
                    "- "
                    + cls._join_fields(
                        [
                            ("tag", hit.get("tag")),
                            ("entity", hit.get("entity")),
                            ("score", cls._score_text(hit.get("score"))),
                            ("evidence", cls._clip(hit.get("evidence"), 420)),
                        ]
                    )
                )

        error_hits = memory_hits.get("error_book", [])
        if error_hits:
            lines.append("Error-book lessons:")
            for hit in error_hits[:2]:
                lines.append(
                    "- "
                    + cls._join_fields(
                        [
                            ("avoid", hit.get("wrong_tag")),
                            ("prefer", hit.get("correct_tag")),
                            ("score", cls._score_text(hit.get("score"))),
                            ("lesson", cls._clip(hit.get("lesson"), 260)),
                            ("evidence", cls._clip(hit.get("evidence"), 320)),
                        ]
                    )
                )

        table_hits = memory_hits.get("table_context_patterns", [])
        if table_hits:
            lines.append("Similar table patterns:")
            for hit in table_hits[:2]:
                lines.append(
                    "- "
                    + cls._join_fields(
                        [
                            ("tag", hit.get("tag")),
                            ("entity", hit.get("entity")),
                            ("score", cls._score_text(hit.get("score"))),
                            ("pattern", cls._clip(hit.get("pattern"), 360)),
                        ]
                    )
                )

        return cls._clip("\n".join(lines), 1200)

    @staticmethod
    def _join_fields(fields: list[tuple[str, Any]]) -> str:
        return "; ".join(f"{key}: {value}" for key, value in fields if value is not None and value != "")

    @staticmethod
    def _clip(value: Any, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _score_text(score: Any) -> str:
        try:
            return f"{float(score):.3f}"
        except (TypeError, ValueError):
            return ""


class ValidatorCorrectorAgent:
    """Agent 2: validates, corrects, and gates LTM writes."""

    def __init__(self, backend: str = "rule", model: str = "meta-llama/Llama-3.2-3B-Instruct") -> None:
        self.backend = backend
        self.llama = LlamaValidator(model) if backend == "llama" else None
        if backend not in {"rule", "llama"}:
            raise ValueError(f"Unsupported validator backend: {backend}")

    def validate(
        self,
        mode: str,
        category: str,
        entity: Any,
        entity_type: str,
        evidence: str,
        selection: Selection,
        candidates: list[dict[str, Any]],
        memory_hits: dict[str, list[dict[str, Any]]],
        gold_tag: str | None,
        attempt: int,
        max_iters: int,
    ) -> ValidationDecision:
        if self.llama is not None:
            llm_decision = self.llama.validate(
                entity=entity,
                entity_type=entity_type,
                evidence=evidence,
                selection=selection,
                candidates=candidates,
                memory_hits=memory_hits,
                gold_tag=None,
            )
            return self._attach_memory_writes(
                llm_decision,
                mode,
                category,
                entity,
                entity_type,
                evidence,
                selection.selected_tag,
                None,
            )

        decision = self._rule_validate(
            mode,
            entity_type,
            selection,
            candidates,
            None,
            attempt,
            max_iters,
        )
        return self._attach_memory_writes(
            decision,
            mode,
            category,
            entity,
            entity_type,
            evidence,
            selection.selected_tag,
            None,
        )

    def supervise_after_loop(
        self,
        mode: str,
        category: str,
        entity: Any,
        entity_type: str,
        evidence: str,
        predicted_tag: str,
        gold_tag: str,
        flags: list[str],
    ) -> ValidationDecision:
        """Use gold after the loop to write supervised memory.

        This method is intentionally separate from ``validate`` so Agent 2 does
        not reveal the gold label to Agent 1 during retry loops.
        """
        if predicted_tag == gold_tag:
            decision = ValidationDecision(
                action="supervise_keep",
                final_tag=predicted_tag,
                passed=True,
                rationale="Post-loop supervision confirmed the final tag.",
                flags=flags,
            )
        else:
            decision = ValidationDecision(
                action="supervise_correct",
                final_tag=gold_tag,
                passed=True,
                rationale="Post-loop supervision corrected the final tag for memory update.",
                flags=flags + ["wrong_against_gold"],
            )
        return self._attach_memory_writes(
            decision=decision,
            mode=mode,
            category=category,
            entity=entity,
            entity_type=entity_type,
            evidence=evidence,
            selected_tag=predicted_tag,
            gold_tag=gold_tag,
        )

    def _rule_validate(
        self,
        mode: str,
        entity_type: str,
        selection: Selection,
        candidates: list[dict[str, Any]],
        gold_tag: str | None,
        attempt: int,
        max_iters: int,
    ) -> ValidationDecision:
        rule_flags = validate_prediction(entity_type, candidates, selection.selected_tag)["flags"]
        top_tag = candidates[0]["tag"] if candidates else ""

        if rule_flags and attempt < max_iters:
            return ValidationDecision(
                action="retry",
                final_tag=selection.selected_tag,
                passed=False,
                rationale="Validator found rule-based risk and requests one retry.",
                feedback_to_selector="The previous selection was low confidence or inconsistent; reconsider candidates."
                + self._risk_feedback(rule_flags),
                flags=rule_flags,
            )

        if mode == "online_without_gt" and self._low_risk(selection, candidates, rule_flags):
            return ValidationDecision(
                action="keep",
                final_tag=selection.selected_tag,
                passed=True,
                rationale="Unsupervised validator accepted a low-risk prediction.",
                flags=rule_flags,
            )

        if rule_flags:
            return ValidationDecision(
                action="flag",
                final_tag=selection.selected_tag or top_tag,
                passed=False,
                rationale="Validator could not safely approve or correct without gold.",
                flags=rule_flags,
            )

        return ValidationDecision(
            action="keep",
            final_tag=selection.selected_tag,
            passed=True,
            rationale="Validator accepted the selector output.",
            flags=rule_flags,
        )

    @staticmethod
    def _low_risk(selection: Selection, candidates: list[dict[str, Any]], flags: list[str]) -> bool:
        if flags or len(candidates) < 2:
            return False
        top_gap = candidates[0]["score"] - candidates[1]["score"]
        return selection.selected_tag == candidates[0]["tag"] and top_gap >= 0.05 and selection.confidence >= 0.95

    @staticmethod
    def _risk_feedback(flags: list[str]) -> str:
        if not flags:
            return ""
        return " Risk signals: " + ", ".join(flags) + "."

    @staticmethod
    def _risk_note(flags: list[str]) -> str:
        if not flags:
            return ""
        return " Risk signals were logged: " + ", ".join(flags) + "."

    def _attach_memory_writes(
        self,
        decision: ValidationDecision,
        mode: str,
        category: str,
        entity: Any,
        entity_type: str,
        evidence: str,
        selected_tag: str,
        gold_tag: str | None,
    ) -> ValidationDecision:
        writes: list[MemoryWrite] = []
        can_write_gold = (
            mode in {"offline_build", "online_with_gt"}
            and gold_tag is not None
            and decision.action in {"keep", "correct", "supervise_keep", "supervise_correct"}
            and decision.passed
        )
        can_write_unsupervised = mode == "online_without_gt" and decision.action == "keep" and decision.passed

        if can_write_gold:
            final_tag = decision.final_tag
            writes.append(self._selector_memory_write(category, entity, entity_type, evidence, final_tag, mode))
            if selected_tag != gold_tag:
                writes.append(self._error_book_write(category, entity, entity_type, evidence, selected_tag, gold_tag, decision))
            if category == "table":
                writes.append(self._table_pattern_write(entity, entity_type, evidence, final_tag, mode))
        elif can_write_unsupervised:
            writes.append(self._selector_memory_write(category, entity, entity_type, evidence, decision.final_tag, mode))
            if category == "table":
                writes.append(self._table_pattern_write(entity, entity_type, evidence, decision.final_tag, mode))

        return ValidationDecision(
            action=decision.action,
            final_tag=decision.final_tag,
            passed=decision.passed,
            rationale=decision.rationale,
            feedback_to_selector=decision.feedback_to_selector,
            memory_writes=writes,
            flags=decision.flags,
        )

    @staticmethod
    def _selector_memory_write(
        category: str,
        entity: Any,
        entity_type: str,
        evidence: str,
        tag: str,
        source: str,
    ) -> MemoryWrite:
        return MemoryWrite(
            namespace="selector_memory",
            payload={
                "source": source,
                "category": category,
                "entity": str(entity),
                "entity_type": entity_type,
                "evidence": evidence,
                "tag": tag,
            },
        )

    @staticmethod
    def _error_book_write(
        category: str,
        entity: Any,
        entity_type: str,
        evidence: str,
        wrong_tag: str,
        correct_tag: str,
        decision: ValidationDecision,
    ) -> MemoryWrite:
        return MemoryWrite(
            namespace="error_book",
            payload={
                "category": category,
                "entity": str(entity),
                "entity_type": entity_type,
                "evidence": evidence,
                "wrong_tag": wrong_tag,
                "correct_tag": correct_tag,
                "reason": decision.rationale,
                "lesson": f"When similar evidence appears, avoid {wrong_tag} and prefer {correct_tag}.",
                "flags": decision.flags,
            },
        )

    @staticmethod
    def _table_pattern_write(entity: Any, entity_type: str, evidence: str, tag: str, source: str) -> MemoryWrite:
        return MemoryWrite(
            namespace="table_context_patterns",
            payload={
                "source": source,
                "entity": str(entity),
                "entity_type": entity_type,
                "pattern": evidence[:600],
                "evidence": evidence,
                "tag": tag,
            },
        )


class LlamaValidator:
    def __init__(self, model: str) -> None:
        self.reranker = LlamaReranker(model)

    def validate(
        self,
        entity: Any,
        entity_type: str,
        evidence: str,
        selection: Selection,
        candidates: list[dict[str, Any]],
        memory_hits: dict[str, list[dict[str, Any]]],
        gold_tag: str | None,
    ) -> ValidationDecision:
        validator_candidates = candidates[:20]
        prompt_evidence = self._validator_evidence(evidence, selection, memory_hits, gold_tag)
        final_tag = self.reranker.choose(entity, entity_type, prompt_evidence, validator_candidates)
        if gold_tag is not None and final_tag != gold_tag:
            return ValidationDecision(
                action="correct",
                final_tag=gold_tag,
                passed=True,
                rationale="LLM validator output differed from gold; gold-supervised correction applied.",
                flags=["wrong_against_gold"],
            )
        action = "keep" if final_tag == selection.selected_tag else "correct"
        return ValidationDecision(
            action=action,
            final_tag=final_tag,
            passed=True,
            rationale=f"LLM validator chose {final_tag}.",
            flags=[],
        )

    @staticmethod
    def _validator_evidence(
        evidence: str,
        selection: Selection,
        memory_hits: dict[str, list[dict[str, Any]]],
        gold_tag: str | None,
    ) -> str:
        memory_summary = selection.memory_context or "No selector memory summary was available."
        gold_note = f"\nGold tag for supervised memory update: {gold_tag}" if gold_tag else ""
        return (
            f"{evidence}\n\n"
            f"Selector proposed: {selection.selected_tag}\n"
            f"Selector rationale: {selection.rationale}\n"
            f"Selector memory summary:\n{memory_summary}" + gold_note
        )
