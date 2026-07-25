#!/usr/bin/env python3
"""T7 efficiency readers (comparing_methods/ags_t7_t28_spec.md, section "T7 -- Efficiency Table").

Per-fact inference cost for six methods, read from the runs that already exist. The spec is
explicit that these are *measured*, not derived from the algorithm ("read them from the run
logs, because retries, fallbacks, and empty-query skips make the real counts differ from the
nominal ones"). Four complications make that instruction impossible to follow literally, so
each one is handled explicitly here and surfaced in metrics.json rather than smoothed over:

1. TWO SCHEMAS. `direct_retrieval` and `one_pass_grounding` were run at commit d0eb0f2,
   before `finalize_candidate_record` gained the `total_llm_calls` / `total_retrieval_calls` /
   `total_*_tokens` / `wall_time` fields (added in 7113bc0). Their traces carry no counters and
   no `rounds` list at all. For those two -- and only those two -- the counts are derived from
   the code path that produced them (`build_candidate_records`,
   run_fintagging_grounding_baseline.py:816-849: exactly one `retrieve_candidates` call, and
   `total_llm_calls=1 if query_mode == "one_pass_grounding" else 0`). Every such number is
   tagged `derived_from_code` in the provenance block; everything else is `logged`.

2. AGS-SEQ CONFLATES REPLAY WITH INFERENCE. `build_ags_seq_method_record`
   (ags_sequential_arms.py:810,852,957) does `calls.extend(revision_calls)` *and*
   `calls.extend(replay_calls)` before `total_llm_calls=len(calls)`, so the persisted field
   is within-episode + counterfactual replay. The spec forbids reporting that as inference
   cost ("do not fold it into the per-fact LLM-call number, or AGS-Seq looks more expensive at
   inference than it is"). The per-call list is separable: every entry carries a `kind` from
   `llm_call_record`, and replay calls are the ones labelled `ags_seq_replay`
   (ags_sequential_arms.py:849). This module recomputes AGS-Seq's inference cost by filtering
   on that field and reports replay separately as adaptation cost.
   Retrieval calls need no such correction -- `pool.extend(new_rounds)` (:865) never receives
   the replay's rounds, so the logged `total_retrieval_calls` is already within-episode only.

3. THE HYPOTHESIS RETRY IS UNCOUNTED. `sample_hypotheses` (ags_frozen_grounding.py:244-265)
   retries once on a parse failure but appends only one call record either way, so
   `len(calls)` is a LOWER BOUND on true model invocations for `frozen_ags` and both
   `ags_seq` arms. A record with `parse_ok=False` is known to have cost two invocations (the
   retry also failed); a record with `parse_ok=True` cost one or two and the log cannot say
   which. Reported as: logged count (primary), a tightened lower bound that adds the known
   +1 per `parse_ok=False` hypothesis call, and the 2x upper bound.

4. A SHARED RERANK CALL SITS OUTSIDE EVERY COUNTER. All six runs were executed with
   RUN_RERANK=1, so each has a `qwen_rerank_predictions.jsonl` with one listwise
   tag-selection LLM call per fact. That call happens after `finalize_candidate_record` has
   already fixed `total_llm_calls`, so it is in NO method's counter. It is uniform across all
   six rows and therefore does not distort their comparison, but it is a real post-generation
   model call and the spec asks us to "flag if any post-generation model call is detected".
   Counted here as its own column (`rerank_llm_calls_per_fact`) rather than folded in.

Wall clock: only the four rich-schema methods have it, and `one_pass_grounding`'s generation
is batched 32-at-a-time (`generate_query_descriptions_vllm`) with no per-fact timer at all, so
the six rows were NOT timed under identical conditions. The spec's own rule for that case is
option (b), drop the column. The measured values are still computed and kept in metrics.json,
marked not-comparable, so the decision is visible rather than silent.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


DEFAULT_RUNS_DIR = _PARENT / "runs_fintagging_grounding_baseline"

# The one call kind that is adaptation cost rather than inference cost (spec: "AGS-Seq's
# counterfactual replay issues extra model calls after the gold label is revealed").
REPLAY_CALL_KIND = "ags_seq_replay"
# Hypothesis sampling is the retry-prone call site (see complication 3 in the module docstring).
HYPOTHESIS_CALL_KIND = "frozen_ags_hypothesis"

LEGACY_SCHEMA = "legacy"
RICH_SCHEMA = "rich"


@dataclass(frozen=True)
class MethodSpec:
    """One row of the efficiency table."""

    label: str
    run_dir: str
    schema: str
    # Only consulted for LEGACY_SCHEMA runs, whose traces carry no counters (complication 1).
    legacy_llm_calls: int = 0
    legacy_retrieval_calls: int = 1
    # LEGACY_SCHEMA only: jsonl holding the generation output, so completion tokens can be
    # recovered by retokenizing rather than estimated from characters (spec forbids estimating).
    legacy_generation_log: str | None = None
    separate_replay: bool = False


# Six rows, in the spec's own order. "Do not list every baseline -- the point is the cost
# *profile* across architecture types (zero-LLM, single-pass, parallel, iterative)."
METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("Direct retrieval", "qwen3_32b_direct_retrieval", LEGACY_SCHEMA, legacy_llm_calls=0),
    MethodSpec(
        "One-pass grounding (free-text)",
        "qwen3_32b_one_pass_grounding",
        LEGACY_SCHEMA,
        legacy_llm_calls=1,
        legacy_generation_log="query_descriptions.jsonl",
    ),
    MethodSpec("Parallel sampling (stochastic, J=2)", "qwen3_32b_parallel_sampling", RICH_SCHEMA),
    MethodSpec("Retrieval-feedback refinement", "qwen3_32b_retrieval_feedback_refinement", RICH_SCHEMA),
    MethodSpec("AGS-Seq (sequential control)", "qwen3_32b_ags_seq", RICH_SCHEMA, separate_replay=True),
    MethodSpec("AGS", "qwen3_32b_frozen_ags", RICH_SCHEMA),
)

# Written last by the pipeline; `grounding_traces.jsonl` is the incremental log the run appends
# to as it goes, so it is the only trace available while a job is still running.
TRACE_CANDIDATES = ("bm25_candidates.jsonl", "grounding_traces.jsonl")
RERANK_LOG = "qwen_rerank_predictions.jsonl"


@dataclass
class FactCost:
    """Inference cost of one fact, with adaptation cost held separately."""

    fact_id: int
    context_id: Any
    modality: str
    llm_calls: float
    retrieval_calls: float
    completion_tokens: float
    wall_time: float | None = None
    # Adaptation cost (AGS-Seq counterfactual replay), excluded from the columns above.
    replay_llm_calls: float = 0.0
    replay_completion_tokens: float = 0.0
    # Known-missing invocations from the uncounted hypothesis retry (complication 3).
    known_retry_calls: float = 0.0
    n_hypothesis_calls: float = 0.0


@dataclass
class MethodCosts:
    spec: MethodSpec
    rows: list[FactCost] = field(default_factory=list)
    call_kinds: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    trace_path: Path | None = None
    rerank_calls: int = 0
    run_complete: bool = True


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_trace(run_path: Path) -> Path | None:
    for name in TRACE_CANDIDATES:
        candidate = run_path / name
        if candidate.exists():
            return candidate
    return None


def gather_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Every logged LLM call for one fact, wherever the producing code path stored it.

    frozen_ags and the ags_seq arms attach a flat top-level list; the free-form methods
    (parallel sampling, retrieval feedback) attach calls to the round that made them.
    """
    for key in ("frozen_ags_llm_calls", "ags_seq_llm_calls"):
        calls = record.get(key)
        if calls:
            return list(calls)
    calls: list[dict[str, Any]] = []
    for round_record in record.get("rounds", []) or []:
        calls.extend(round_record.get("llm_calls") or [])
    return calls


def count_tokens(texts: list[str], tokenizer: Any) -> list[int]:
    if tokenizer is None:
        return [0] * len(texts)
    return [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]


def load_tokenizer(model_name: str) -> Any | None:
    """The generation-time tokenizer, for the two legacy runs whose logs never stored a
    completion-token count. `QueryGenerator.count_text_tokens`
    (run_fintagging_grounding_baseline.py:1319) counts exactly this way, so retokenizing
    reproduces the number the rich-schema runs recorded rather than approximating it."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as error:  # noqa: BLE001 - tokenizer is optional; report and continue.
        print(f"WARNING: could not load tokenizer {model_name!r}: {error}", flush=True)
        return None


def count_rerank_calls(run_path: Path, limit: int | None = None) -> int:
    """The shared listwise rerank stage: one model call per fact, in no method's own counter
    (complication 4)."""
    path = run_path / RERANK_LOG
    if not path.exists():
        return 0
    total = 0
    for _ in stream_jsonl(path):
        total += 1
        if limit is not None and total >= limit:
            break
    return total


def read_legacy(spec: MethodSpec, run_path: Path, tokenizer: Any, limit: int | None) -> MethodCosts:
    """Pre-7113bc0 traces: no counters, no `rounds`. Counts come from the code path that wrote
    them (`build_candidate_records`), completion tokens from retokenizing the generation log."""
    out = MethodCosts(spec=spec)
    trace = find_trace(run_path)
    if trace is None:
        out.notes.append(f"no trace file found under {run_path}")
        return out
    out.trace_path = trace

    completion_by_fact: dict[int, int] = {}
    if spec.legacy_generation_log:
        gen_path = run_path / spec.legacy_generation_log
        if gen_path.exists():
            raw_by_fact = {
                int(row["example_idx"]): str(row.get("raw_output", "") or "")
                for row in stream_jsonl(gen_path)
            }
            fact_ids = sorted(raw_by_fact)
            counts = count_tokens([raw_by_fact[fact_id] for fact_id in fact_ids], tokenizer)
            completion_by_fact = dict(zip(fact_ids, counts))
            if tokenizer is None:
                out.notes.append(
                    f"{spec.legacy_generation_log} has raw_output but no completion_tokens field, "
                    "and no tokenizer was loaded -- completion tokens reported as 0, not measured"
                )
                out.provenance["completion_tokens_per_fact"] = "unavailable"
            else:
                out.provenance["completion_tokens_per_fact"] = "retokenized_from_raw_output"
        else:
            out.notes.append(f"missing generation log {gen_path}")
    else:
        out.provenance["completion_tokens_per_fact"] = "derived_from_code (no LLM call)"

    out.provenance["llm_calls_per_fact"] = "derived_from_code (build_candidate_records)"
    out.provenance["retrieval_calls_per_fact"] = "derived_from_code (one retrieve_candidates call)"
    out.provenance["wallclock_sec_per_fact"] = "unavailable (pre-dates wall_time logging)"
    out.notes.append(
        "trace written before finalize_candidate_record logged per-fact counters; LLM and "
        "retrieval call counts are derived from the code path, not read from the log"
    )

    for record in stream_jsonl(trace):
        if limit is not None and len(out.rows) >= limit:
            break
        fact_id = int(record["example_idx"])
        out.rows.append(
            FactCost(
                fact_id=fact_id,
                context_id=record.get("context_id"),
                modality=record.get("input_type", ""),
                llm_calls=float(spec.legacy_llm_calls),
                retrieval_calls=float(spec.legacy_retrieval_calls),
                completion_tokens=float(completion_by_fact.get(fact_id, 0)),
                wall_time=None,
            )
        )
    if spec.legacy_llm_calls:
        out.call_kinds["query_description (batched)"] = len(out.rows)
    return out


def read_rich(spec: MethodSpec, run_path: Path, limit: int | None) -> MethodCosts:
    """Post-7113bc0 traces: per-fact counters are logged. Only AGS-Seq needs correction, to
    strip counterfactual-replay calls back out of them (complication 2)."""
    out = MethodCosts(spec=spec)
    trace = find_trace(run_path)
    if trace is None:
        out.notes.append(f"no trace file found under {run_path}")
        return out
    out.trace_path = trace
    if trace.name != "bm25_candidates.jsonl":
        out.run_complete = False
        out.notes.append(
            f"reading {trace.name}: the run's final bm25_candidates.jsonl does not exist yet, "
            "so this row covers only the facts written so far"
        )

    out.provenance["llm_calls_per_fact"] = "logged"
    out.provenance["retrieval_calls_per_fact"] = "logged (total_retrieval_calls)"
    out.provenance["completion_tokens_per_fact"] = "logged (per-call completion_tokens)"
    out.provenance["wallclock_sec_per_fact"] = "logged (per-fact time.monotonic delta)"

    for record in stream_jsonl(trace):
        if limit is not None and len(out.rows) >= limit:
            break
        calls = gather_calls(record)
        for call in calls:
            kind = str(call.get("kind", "unknown"))
            out.call_kinds[kind] = out.call_kinds.get(kind, 0) + 1

        inference_calls = calls
        replay_calls: list[dict[str, Any]] = []
        if spec.separate_replay:
            replay_calls = [call for call in calls if call.get("kind") == REPLAY_CALL_KIND]
            inference_calls = [call for call in calls if call.get("kind") != REPLAY_CALL_KIND]

        def completion_of(items: list[dict[str, Any]]) -> float:
            return float(sum(int(call.get("completion_tokens", 0) or 0) for call in items))

        hypothesis_calls = [call for call in calls if call.get("kind") == HYPOTHESIS_CALL_KIND]
        # A hypothesis call logged parse_ok=False definitely cost a second, unlogged
        # invocation (ags_frozen_grounding.py:247-250 retries, then appends one record).
        known_retry = float(sum(1 for call in hypothesis_calls if not call.get("parse_ok", True)))

        if calls:
            llm_calls = float(len(inference_calls))
            completion_tokens = completion_of(inference_calls)
        else:
            # No per-call list on this record; fall back to the record-level counters. For a
            # separate_replay method those totals are replay-contaminated and cannot be split,
            # so leave them out rather than report a number the spec explicitly rejects.
            if spec.separate_replay:
                continue
            llm_calls = float(record.get("total_llm_calls", 0) or 0)
            completion_tokens = float(record.get("total_completion_tokens", 0) or 0)

        wall_time = record.get("wall_time")
        out.rows.append(
            FactCost(
                fact_id=int(record["example_idx"]),
                context_id=record.get("context_id"),
                modality=record.get("input_type", ""),
                llm_calls=llm_calls,
                retrieval_calls=float(record.get("total_retrieval_calls", 0) or 0),
                completion_tokens=completion_tokens,
                wall_time=None if wall_time is None else float(wall_time),
                replay_llm_calls=float(len(replay_calls)),
                replay_completion_tokens=completion_of(replay_calls),
                known_retry_calls=known_retry,
                n_hypothesis_calls=float(len(hypothesis_calls)),
            )
        )

    if spec.separate_replay:
        out.notes.append(
            f"logged total_llm_calls folds counterfactual replay into inference; recomputed "
            f"here from the per-call {REPLAY_CALL_KIND!r} tag. Replay reported separately as "
            "adaptation cost, per the spec's 'separate inference cost from adaptation cost'."
        )
    return out


def read_method(
    spec: MethodSpec,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    tokenizer: Any = None,
    limit: int | None = None,
) -> MethodCosts:
    run_path = runs_dir / spec.run_dir
    if spec.schema == LEGACY_SCHEMA:
        out = read_legacy(spec, run_path, tokenizer, limit)
    else:
        out = read_rich(spec, run_path, limit)
    out.rerank_calls = count_rerank_calls(run_path, limit)
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def summarize(costs: MethodCosts, report_wallclock: bool, hardware_note: str, batch_note: str) -> dict[str, Any]:
    """One efficiency.csv row."""
    rows = costs.rows
    n_facts = len(rows)
    timed = [row.wall_time for row in rows if row.wall_time is not None]
    wallclock = _mean([float(value) for value in timed]) if report_wallclock else None

    summary: dict[str, Any] = {
        "method": costs.spec.label,
        "n_facts": n_facts,
        "llm_calls_per_fact": _mean([row.llm_calls for row in rows]),
        "retrieval_calls_per_fact": _mean([row.retrieval_calls for row in rows]),
        "completion_tokens_per_fact": _mean([row.completion_tokens for row in rows]),
        "wallclock_sec_per_fact": wallclock,
        # Kept out of llm_calls_per_fact on purpose: uniform across all six rows, applied after
        # each method's own counter was finalized (complication 4).
        "rerank_llm_calls_per_fact": (
            round(costs.rerank_calls / n_facts, 6) if n_facts and costs.rerank_calls else 0.0
        ),
        # Adaptation, not inference -- AGS-Seq only, zero elsewhere.
        "replay_llm_calls_per_fact": _mean([row.replay_llm_calls for row in rows]),
        "replay_completion_tokens_per_fact": _mean([row.replay_completion_tokens for row in rows]),
        "hardware_note": hardware_note,
        "batch_note": batch_note,
    }

    known_retry = sum(row.known_retry_calls for row in rows)
    hypothesis_calls = sum(row.n_hypothesis_calls for row in rows)
    if hypothesis_calls:
        # The logged count is a lower bound: an unknown share of parse_ok=True hypothesis calls
        # also retried once (module docstring, complication 3).
        summary["llm_calls_per_fact_lower_bound"] = round(
            (sum(row.llm_calls for row in rows) + known_retry) / n_facts, 6
        )
        summary["llm_calls_per_fact_upper_bound"] = round(
            (sum(row.llm_calls for row in rows) + hypothesis_calls) / n_facts, 6
        )
    else:
        summary["llm_calls_per_fact_lower_bound"] = summary["llm_calls_per_fact"]
        summary["llm_calls_per_fact_upper_bound"] = summary["llm_calls_per_fact"]
    return summary


def diagnostics(costs: MethodCosts) -> dict[str, Any]:
    rows = costs.rows
    modalities = sorted({row.modality for row in rows if row.modality})
    per_modality = {
        modality: {
            "n_facts": sum(1 for row in rows if row.modality == modality),
            "retrieval_calls_per_fact": _mean(
                [row.retrieval_calls for row in rows if row.modality == modality]
            ),
            "llm_calls_per_fact": _mean([row.llm_calls for row in rows if row.modality == modality]),
        }
        for modality in modalities
    }
    return {
        "trace": str(costs.trace_path) if costs.trace_path else None,
        "run_complete": costs.run_complete,
        "n_facts": len(rows),
        "call_kinds": dict(sorted(costs.call_kinds.items())),
        "provenance": costs.provenance,
        "notes": costs.notes,
        "rerank_stage_calls": costs.rerank_calls,
        # The spec's own check on AGS's retrieval column: "AGS on table facts issues 2 per
        # hypothesis (def + lab) = 4; on text 1 per hypothesis = 2. The reported mean is over
        # the real table/text mix, so it will be between 2 and 4, not exactly 4."
        "by_modality": per_modality,
        "measured_wallclock_sec_per_fact": _mean(
            [float(row.wall_time) for row in rows if row.wall_time is not None]
        ),
        "facts_with_wallclock": sum(1 for row in rows if row.wall_time is not None),
    }
