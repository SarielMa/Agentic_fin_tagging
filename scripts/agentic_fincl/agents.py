# Defines the retriever, selector, validator, and local LLM helpers for the baseline.
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .data import Example, TaxonomyConcept
from .hf_generation import load_local_causal_lm, model_input_device, move_inputs_to_device
from .memory_store import LTMStore
from .retrieval import TaxonomyBM25, rank_memory


TAG_RE = re.compile(r"us-gaap:[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Selection:
    tag: str
    rationale: str
    raw_output: str
    correct_memories: list[dict[str, Any]]
    error_memories: list[dict[str, Any]]


@dataclass(frozen=True)
class ValidationFeedback:
    suggested_tag: str
    feedback: str
    raw_output: str
    error_memories: list[dict[str, Any]]


@dataclass(frozen=True)
class GTCoaching:
    """Critic feedback produced during memory-build, where ground truth is known."""

    hint: str
    suggested_tag: str
    raw_output: str
    coach_mode: str


@dataclass(frozen=True)
class MemoryWriteResult:
    book: str
    comment: str
    predicted_tag: str
    correct_tag: str


class LocalLLM:
    """Small deterministic wrapper around a local Hugging Face causal LM."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.torch = torch
        self.tokenizer, self.model, _ = load_local_causal_lm(
            model_name,
            torch,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        # Qwen3-14B has a 40,960-token native context. The old 8,192 default
        # truncated the prompt from the right, cutting the context block and the
        # validator feedback (both appended after the 200-candidate list) so the
        # selector chose blind. Default to 32,768 to keep the full prompt, with
        # headroom under the model limit for generation.
        self.max_input_tokens = int(os.environ.get("FINAI_MAX_INPUT_TOKENS", "32768"))

    def generate(self, system: str, user: str, max_new_tokens: int = 128) -> str:
        prompt = self._prompt(system, user)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = move_inputs_to_device(inputs, model_input_device(self.model))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with self.torch.inference_mode():
            # Fixed testing uses greedy decoding. Temperature/top_p are sampling-only;
            # neutral values prevent inherited model sampling config warnings.
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    def _prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class RetrieverAgent:
    """Agent 1: type-filtered BM25 over taxonomy concept.text."""

    def __init__(self, taxonomy: list[TaxonomyConcept]) -> None:
        self.index = TaxonomyBM25(taxonomy)

    def retrieve(self, context: str, entity_type: str, top_k: int = 200) -> list[dict[str, Any]]:
        return self.index.retrieve(context=context, entity_type=entity_type, top_k=top_k)


class SelectorAgent:
    """Agent 2: selects one tag using candidates plus correct/error LTM memory."""

    def __init__(self, model_name: str, memory_k: int = 5) -> None:
        self.llm = LocalLLM(model_name)
        self.memory_k = memory_k

    def select(
        self,
        example: Example,
        context: str,
        candidates: list[dict[str, Any]],
        ltm: LTMStore,
        feedback: str = "",
    ) -> Selection:
        correct_memories = rank_memory(ltm.records("correct_book"), context, example.entity_type, self.memory_k)
        error_memories = rank_memory(ltm.records("error_book"), context, example.entity_type, self.memory_k)

        raw_output = self.llm.generate(
            system=(
                "You are Agent 2, the selector. Select exactly one US-GAAP tag from the "
                "candidate list. Use the correct book as positive examples and the error "
                "book as warnings. Return compact JSON with keys tag and rationale."
            ),
            user=_selection_prompt(
                example=example,
                context=context,
                candidates=candidates,
                correct_memories=correct_memories,
                error_memories=error_memories,
                feedback=feedback,
            ),
            max_new_tokens=128,
        )
        tag = _extract_candidate_tag(raw_output, candidates) or _first_candidate_tag(candidates)
        rationale = _extract_field(raw_output, "rationale") or _clip(raw_output, 240)
        return Selection(
            tag=tag,
            rationale=rationale,
            raw_output=raw_output,
            correct_memories=correct_memories,
            error_memories=error_memories,
        )


class ValidatorAgent:
    """Agent 3: validates predictions and is the only writer of LTM memory."""

    def __init__(self, model_name: str, memory_k: int = 5) -> None:
        self.llm = LocalLLM(model_name)
        self.memory_k = memory_k

    def review_without_gt(
        self,
        example: Example,
        context: str,
        candidates: list[dict[str, Any]],
        selector_tag: str,
        ltm: LTMStore,
    ) -> ValidationFeedback:
        # Prediction-aware error retrieval, two steps:
        #   1. keep only past errors whose predicted_tag matches the selector's current pick;
        #   2. rank those survivors by context BM25.
        # If step 1 leaves nothing, error_memories is empty and the validator falls back to
        # its own knowledge (the prompt renders the error book as "(none)").
        same_pred_errors = [
            record for record in ltm.records("error_book")
            if record.get("predicted_tag") == selector_tag
        ]
        error_memories = rank_memory(same_pred_errors, context, example.entity_type, self.memory_k)
        raw_output = self.llm.generate(
            system=(
                "You are Agent 3, the validator in testing mode. You cannot see ground truth. "
                "Independently select the best candidate tag using only the context, candidate "
                "list, and error book. Return compact JSON with keys tag and feedback."
            ),
            user=_validator_prompt(
                example=example,
                context=context,
                candidates=candidates,
                selector_tag=selector_tag,
                error_memories=error_memories,
            ),
            max_new_tokens=128,
        )
        suggested_tag = _extract_candidate_tag(raw_output, candidates) or selector_tag
        feedback = _extract_field(raw_output, "feedback") or _clip(raw_output, 240)
        return ValidationFeedback(
            suggested_tag=suggested_tag,
            feedback=feedback,
            raw_output=raw_output,
            error_memories=error_memories,
        )

    def coach_with_gt(
        self,
        example: Example,
        context: str,
        candidates: list[dict[str, Any]],
        selector_tag: str,
        tried_tags: list[str] | None = None,
        coach_mode: str = "hint",
    ) -> GTCoaching:
        """Memory-build critic: the student was wrong, so use ground truth to coach a retry.

        coach_mode="hint": describe the correct concept family and why the prediction is
        wrong, WITHOUT revealing the literal ground-truth tag (no test-time oracle leak).
        coach_mode="oracle": reveal the ground-truth tag directly (for A/B comparison).

        ``tried_tags`` lists every candidate the student has already picked and had
        rejected; passing it lets successive hints escalate instead of repeating.
        """
        if example.answer is None:
            raise ValueError("coach_with_gt requires a ground-truth answer.")

        if coach_mode == "oracle":
            hint = f"The correct tag is {example.answer}. Re-select it from the candidate list."
            return GTCoaching(hint=hint, suggested_tag=example.answer, raw_output="", coach_mode=coach_mode)

        rejected = ", ".join(dict.fromkeys(tried_tags or [selector_tag]))
        raw_output = self.llm.generate(
            system=(
                "You are Agent 3, the validator in memory-build mode. You can see the "
                "ground-truth tag but must NOT reveal it verbatim. The selector keeps picking "
                "wrong tags. Write a short corrective hint that points to the correct concept "
                "family and explains why the already-rejected tags are wrong, so the selector "
                "can find the right candidate itself. Do not repeat a rejected tag. Return "
                "compact JSON with keys hint and family."
            ),
            user=(
                f"Entity type: {example.entity_type}\n"
                f"Entity value: {example.entity}\n"
                f"Already-rejected tags (all wrong): {rejected}\n"
                f"Ground-truth tag (do not quote it): {example.answer}\n"
                "Candidate tags:\n"
                f"{_format_candidates(candidates)}\n"
                "Context:\n"
                f"{_clip(context, 2200)}\n"
                'Return JSON only, for example: {"hint": "...", "family": "..."}'
            ),
            max_new_tokens=128,
        )
        hint = _extract_field(raw_output, "hint") or _clip(raw_output, 240)
        if not hint:
            hint = f"{rejected} are wrong; reconsider the candidate that matches this line item."
        return GTCoaching(hint=hint, suggested_tag="", raw_output=raw_output, coach_mode=coach_mode)

    def write_with_gt(
        self,
        example: Example,
        context: str,
        selector_tag: str,
        ltm: LTMStore,
    ) -> MemoryWriteResult:
        if example.answer is None:
            raise ValueError("Memory-build mode requires a ground-truth answer.")

        correct = selector_tag == example.answer
        comment = self._memory_comment(example, context, selector_tag, correct)
        if correct:
            ltm.write_correct(
                entity_type=example.entity_type,
                value=str(example.entity),
                context=context,
                tag=selector_tag,
                comment=comment,
            )
            book = "correct_book"
        else:
            ltm.write_error(
                entity_type=example.entity_type,
                value=str(example.entity),
                context=context,
                predicted_tag=selector_tag,
                correct_tag=example.answer,
                comment=comment,
            )
            book = "error_book"

        return MemoryWriteResult(
            book=book,
            comment=comment,
            predicted_tag=selector_tag,
            correct_tag=example.answer,
        )

    def write_fallback_with_gt(
        self,
        example: Example,
        context: str,
        selector_tag: str,
        ltm: LTMStore,
    ) -> MemoryWriteResult:
        """Student failed after all coached retries. Bank the ground-truth (label -> tag)
        exemplar in correct_book regardless (its value is the mapping, not whether the
        student earned it) AND record the contrastive miss in error_book."""
        if example.answer is None:
            raise ValueError("Memory-build mode requires a ground-truth answer.")

        comment = self._memory_comment(example, context, selector_tag, correct=False)
        ltm.write_correct(
            entity_type=example.entity_type,
            value=str(example.entity),
            context=context,
            tag=example.answer,
            comment=comment,
        )
        ltm.write_error(
            entity_type=example.entity_type,
            value=str(example.entity),
            context=context,
            predicted_tag=selector_tag,
            correct_tag=example.answer,
            comment=comment,
        )
        return MemoryWriteResult(
            book="correct_book+error_book",
            comment=comment,
            predicted_tag=selector_tag,
            correct_tag=example.answer,
        )

    def _memory_comment(self, example: Example, context: str, selector_tag: str, correct: bool) -> str:
        verdict = "correct" if correct else "wrong"
        raw_output = self.llm.generate(
            system=(
                "You are Agent 3, the validator in memory-build mode. You can see the "
                "ground-truth tag. Write one concise lesson for future selection."
            ),
            user=(
                f"Entity type: {example.entity_type}\n"
                f"Entity value: {example.entity}\n"
                f"Selector prediction: {selector_tag}\n"
                f"Ground truth: {example.answer}\n"
                f"Verdict: {verdict}\n"
                f"Context: {_clip(context, 1800)}\n\n"
                "Return one short sentence explaining why this memory should guide future cases."
            ),
            max_new_tokens=80,
        )
        comment = _clip(raw_output, 320)
        if comment:
            return comment
        if correct:
            return "The selected tag matches the ground truth for this entity type and context."
        return f"The selector predicted {selector_tag}, but the ground truth is {example.answer}."


def _selection_prompt(
    example: Example,
    context: str,
    candidates: list[dict[str, Any]],
    correct_memories: list[dict[str, Any]],
    error_memories: list[dict[str, Any]],
    feedback: str,
) -> str:
    parts = [
        f"Entity type: {example.entity_type}",
        f"Entity value: {example.entity}",
        "Candidate tags:",
        _format_candidates(candidates),
        "Correct book memories:",
        _format_memories("correct_book", correct_memories),
        "Error book memories:",
        _format_memories("error_book", error_memories),
    ]
    if feedback:
        parts.extend(["Validator feedback from prior attempt:", feedback])
    parts.extend(
        [
            "Context:",
            _clip(context, 2200),
            'Return JSON only, for example: {"tag": "us-gaap:ExampleTag", "rationale": "..."}',
        ]
    )
    return "\n".join(parts)


def _validator_prompt(
    example: Example,
    context: str,
    candidates: list[dict[str, Any]],
    selector_tag: str,
    error_memories: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            f"Entity type: {example.entity_type}",
            f"Entity value: {example.entity}",
            f"Selector prediction: {selector_tag}",
            "Candidate tags:",
            _format_candidates(candidates),
            "Error book memories:",
            _format_memories("error_book", error_memories),
            "Context:",
            _clip(context, 2200),
            'Return JSON only, for example: {"tag": "us-gaap:ExampleTag", "feedback": "..."}',
        ]
    )


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "(none)"
    lines = []
    for candidate in candidates:
        lines.append(
            f"{candidate['rank']}. {candidate['tag']} | {_clip(candidate.get('text', ''), 90)}"
        )
    return "\n".join(lines)


def _format_memories(book: str, memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(none)"
    lines = []
    for memory in memories:
        if book == "correct_book":
            lines.append(
                "- "
                + "; ".join(
                    [
                        f"value={_clip(memory.get('value', ''), 90)}",
                        f"tag={memory.get('tag')}",
                        f"comment={_clip(memory.get('comment', ''), 180)}",
                        f"context={_clip(memory.get('context', ''), 220)}",
                    ]
                )
            )
        else:
            lines.append(
                "- "
                + "; ".join(
                    [
                        f"value={_clip(memory.get('value', ''), 90)}",
                        f"predicted={memory.get('predicted_tag')}",
                        f"correct={memory.get('correct_tag')}",
                        f"comment={_clip(memory.get('comment', ''), 180)}",
                        f"context={_clip(memory.get('context', ''), 220)}",
                    ]
                )
            )
    return "\n".join(lines)


def _extract_candidate_tag(text: str, candidates: list[dict[str, Any]]) -> str:
    candidate_tags = {candidate["tag"] for candidate in candidates}
    suffix_to_tag = {tag.split(":", 1)[-1]: tag for tag in candidate_tags}

    for match in TAG_RE.findall(text or ""):
        if match in candidate_tags:
            return match

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text or "")
    for token in tokens:
        tag = suffix_to_tag.get(token)
        if tag is not None:
            return tag
    return ""


def _extract_field(text: str, field: str) -> str:
    parsed = _extract_json(text)
    if isinstance(parsed, dict):
        value = parsed.get(field)
        if value is not None:
            return _clip(value, 500)
    pattern = re.compile(rf'"?{re.escape(field)}"?\s*:\s*"([^"]+)"', flags=re.I)
    match = pattern.search(text or "")
    if match:
        return _clip(match.group(1), 500)
    return ""


def _extract_json(text: str) -> Any:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _first_candidate_tag(candidates: list[dict[str, Any]]) -> str:
    return str(candidates[0]["tag"]) if candidates else ""


def _clip(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
