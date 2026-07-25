#!/usr/bin/env python3
"""Row 3.9: + LLM verification layer (ags_table5_ablation_spec.md section 3.9). GPU, vLLM.

The only row requiring new generation: J=2 calls per fact (~2 x 2,509 = 5,020) re-judging
FAMILY/ROLE/EVENT on the top M=10 cluster-representative candidates from AGS's own fused
ranking. QUALIFIER/SCOPE/TEMPORAL, and FAMILY/ROLE/EVENT outside the top M, stay symbolic --
core.hybrid_agree_score enforces that split; this script only produces the verdicts file
core.py's `llm_verifier_verdicts` hook consumes.

Standalone by design: it does not import or extend run_fintagging_grounding_baseline.py's
--query-mode dispatch (build_comparison_candidate_records), which other experiments run
against concurrently. It reads AGS's already-computed test-split trace, reuses the prompt-
building primitives (serialize_evidence, format_candidate_for_prompt, QueryGenerator,
parse_json_object) as pure, read-only imports, and writes only to its own output file.

The verifier never sees the gold concept -- nothing in the prompt below includes it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_sequential_arms import cluster_representatives  # noqa: E402
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from ags_table5_ablation.data_prep import DEFAULT_TEST_TRACE, stream_jsonl  # noqa: E402
from run_fintagging_grounding_baseline import (  # noqa: E402
    QueryGenerator,
    build_prompt_under_query_budget,
    format_candidate_for_prompt,
    llm_call_record,
    normalize_tag,
    parse_json_object,
    serialize_evidence,
)


VERIFIER_DIMENSIONS = ("FAMILY", "ROLE", "EVENT")
VERDICT_TO_MATCHED = {"support": True, "contradict": False, "unresolved": None}


class _EvidenceView:
    """serialize_evidence needs an Example; the trace only persists the flattened fields it
    already rendered. This adapts the persisted record without re-deriving anything."""

    __slots__ = ("row_context", "column_context", "query_context", "input_type", "entity", "entity_type")

    def __init__(self, record: dict[str, Any]) -> None:
        self.row_context = record.get("row_context", "")
        self.column_context = record.get("column_context", "")
        self.query_context = record.get("query_context", "")
        self.input_type = record.get("input_type", "")
        self.entity = record.get("entity", "")
        self.entity_type = record.get("type", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b")
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--cluster-scan-depth", type=int, default=60)
    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--query-generation-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    # THIS is the cap that bounds the verdict JSON: QueryGenerator.generate_many passes
    # query_max_new_tokens as vLLM's max_tokens (run_fintagging_grounding_baseline.py:1331).
    # One verdict entry is ~110-130 tokens (tag + 3 dimensions + confidence + punctuation), so
    # top_m=10 needs ~1,100-1,300 plus the wrapper. The original 512 truncated EVERY response
    # mid-array: a full 2,369-call run produced a 100% parse failure rate and zero verdicts.
    # Sized at 1536 to leave headroom; raise it alongside --top-m if that is ever increased.
    parser.add_argument("--query-max-new-tokens", type=int, default=1536)
    # load_vllm_engine (run_fintagging_grounding_baseline.py) sizes max_model_len off the
    # plain (non-query-prefixed) max_input_tokens/max_new_tokens as well; both are read even
    # though this script only ever calls generate_one on the query_* generation path.
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--candidate-doc-max-chars", type=int, default=320)
    parser.add_argument("--query-temperature", type=float, default=0.0)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    # Fail-fast. The truncation bug ran 4h34m across 2,369 calls at a 0% parse rate before
    # anyone looked at the output; a zero firing rate is also the *expected* result of genuine
    # abstention, so nothing downstream flags it. Check early and abort loudly instead.
    parser.add_argument("--abort-check-after", type=int, default=25, help="0 disables the guard.")
    parser.add_argument("--abort-if-parse-rate-below", type=float, default=0.5)
    return parser.parse_args()


def build_verifier_messages(
    record: dict[str, Any],
    hypothesis: dict[str, Any],
    representatives: list[dict[str, Any]],
    context_max_chars: int,
    doc_max_chars: int,
) -> list[dict[str, str]]:
    evidence = serialize_evidence(_EvidenceView(record), context_max_chars)
    candidate_text = "\n\n".join(
        format_candidate_for_prompt(candidate, doc_max_chars) for candidate in representatives
    )
    dims = hypothesis.get("dimensions", {})
    hypothesis_text = json.dumps(
        {dimension: dims.get(dimension, dims.get(dimension.lower(), "UNRESOLVED")) for dimension in VERIFIER_DIMENSIONS},
        ensure_ascii=False,
    )
    user = f"""Judge how well each candidate taxonomy concept matches a grounding hypothesis, on exactly three
dimensions: FAMILY (broad accounting domain), ROLE (specific function), EVENT (event or state).

Do not judge any other dimension. Do not try to identify which candidate is the correct answer -- assess
each dimension independently for every candidate. If the evidence does not let you decide a dimension for
a candidate, say "unresolved" rather than guessing.

Hypothesis (FAMILY/ROLE/EVENT only):
{hypothesis_text}

Evidence:
{evidence}

Candidates:
{candidate_text}

Return JSON only, one entry per candidate tag, with this schema:
{{"verdicts": [{{"tag": "us-gaap:Example", "FAMILY": "support|contradict|unresolved", "ROLE": "support|contradict|unresolved", "EVENT": "support|contradict|unresolved", "confidence": 0.0}}]}}"""
    return [
        {"role": "system", "content": "You verify US-GAAP grounding hypotheses against candidate concepts on FAMILY/ROLE/EVENT only."},
        {"role": "user", "content": user},
    ]


def salvage_verdict_entries(raw_output: str) -> list[dict[str, Any]]:
    """Recover the complete objects from a truncated `{"verdicts": [ {...}, {...}, {...`.

    A verdict list is independent per candidate, so a response cut off mid-array still carries
    usable judgements for every candidate that finished. Whole-document json.loads throws all
    of them away because the array never closes. This scans the array and keeps each balanced
    `{...}`, stopping at the point of truncation.

    Defence in depth, not the primary fix -- --query-max-new-tokens is now sized so responses
    complete. Any salvage is counted and reported so silent partial coverage stays visible.
    """
    text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE)
    anchor = text.find('"verdicts"')
    start = text.find("[", anchor if anchor >= 0 else 0)
    if start < 0:
        return []

    entries: list[dict[str, Any]] = []
    depth = 0
    in_string = False
    escape = False
    entry_start = -1
    for index in range(start + 1, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                entry_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and entry_start >= 0:
                try:
                    parsed = json.loads(text[entry_start : index + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        entries.append(parsed)
                entry_start = -1
        elif char == "]" and depth == 0:
            break
    return entries


def parse_verifier_output(
    raw_output: str, candidate_tags: list[str]
) -> tuple[dict[str, dict[str, Any]], bool, str]:
    """Returns (verdicts_by_tag, parse_ok, parse_mode).

    parse_mode is "clean" (whole document parsed), "salvaged" (recovered from a truncated
    array) or "failed" (nothing usable), so a run can be audited for silent truncation rather
    than only for total failure.
    """
    parsed, whole_ok = parse_json_object(raw_output)
    entries = parsed.get("verdicts") if isinstance(parsed, dict) else None
    mode = "clean"
    if not isinstance(entries, list) or not entries:
        entries = salvage_verdict_entries(raw_output)
        mode = "salvaged" if entries else "failed"

    verdicts_by_tag: dict[str, dict[str, Any]] = {}
    allowed = set(candidate_tags)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tag = normalize_tag(entry.get("tag", ""))
        if tag not in allowed:
            continue
        verdict: dict[str, Any] = {}
        for dimension in VERIFIER_DIMENSIONS:
            raw_verdict = str(entry.get(dimension, "")).strip().lower()
            verdict[dimension] = VERDICT_TO_MATCHED.get(raw_verdict, None)
        verdict["confidence"] = float(entry.get("confidence", 0.0) or 0.0)
        verdicts_by_tag[tag] = verdict

    if not verdicts_by_tag:
        return {}, False, "failed"
    if mode == "clean" and not whole_ok:
        mode = "salvaged"
    return verdicts_by_tag, True, mode


def load_existing(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    """Reusable rows only: a row is reusable if it actually carries verdicts.

    Resuming must NOT treat a failed call as done. The truncation bug produced 2,369 rows with
    parse_ok=false and empty verdicts_by_tag; keying resume purely on (fact_id, hypothesis_idx)
    would skip every one of them forever and silently bake the failure into the final table.
    Failed rows are counted and reported, then regenerated.
    """
    existing: dict[tuple[int, int], dict[str, Any]] = {}
    skipped = 0
    if not path.exists():
        return existing, skipped
    for row in stream_jsonl(path):
        key = (int(row["fact_id"]), int(row["hypothesis_idx"]))
        if row.get("parse_ok") and row.get("verdicts_by_tag"):
            existing[key] = row
        else:
            skipped += 1
            existing.pop(key, None)
    return existing, skipped


def main() -> None:
    args = parse_args()
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_calls_path = args.output_dir / "llm_verifier_calls.jsonl"
    verdicts_path = args.output_dir / "llm_verifier_verdicts.json"

    existing, unusable_existing = load_existing(raw_calls_path) if args.resume else ({}, 0)
    if args.resume:
        print(
            f"Resume: {len(existing)} reusable calls, {unusable_existing} previous calls had no "
            "verdicts and will be regenerated.",
            flush=True,
        )
    firing_counts = Counter()
    opportunity_counts = Counter()
    parse_modes = Counter()
    generated = 0
    all_verdicts: list[dict[str, Any]] = []
    generator: QueryGenerator | None = None

    handle = raw_calls_path.open("a" if (args.resume and raw_calls_path.exists()) else "w", encoding="utf-8")
    try:
        for offset, record in enumerate(stream_jsonl(args.test_trace), start=1):
            if args.limit is not None and offset > args.limit:
                break
            fact_id = int(record["example_idx"])
            ranking = record.get("final_candidates") or record.get("candidates") or []
            representatives = cluster_representatives(ranking, normalization_map, args.top_m, args.cluster_scan_depth)
            candidate_tags = [normalize_tag(candidate["tag"]) for candidate in representatives]

            for hypothesis in record.get("frozen_ags_hypotheses", []):
                hyp_idx = int(hypothesis["hypothesis_idx"])
                key = (fact_id, hyp_idx)
                if key in existing:
                    row = existing[key]
                else:
                    if generator is None:
                        generator = QueryGenerator(args)
                    prompt, prompt_tokens, used_context_chars = build_prompt_under_query_budget(
                        generator.tokenizer,
                        lambda ctx_chars: build_verifier_messages(
                            record, hypothesis, representatives, ctx_chars, args.candidate_doc_max_chars
                        ),
                        context_max_chars=args.query_context_max_chars,
                        max_input_tokens=args.query_max_input_tokens,
                    )
                    raw_output = generator.generate_one(prompt)
                    verdicts_by_tag, parse_ok, parse_mode = parse_verifier_output(
                        raw_output, candidate_tags
                    )
                    completion_tokens = generator.count_text_tokens(raw_output)
                    generated += 1
                    parse_modes[parse_mode] += 1
                    call = llm_call_record(
                        "table5_llm_verifier",
                        raw_output=raw_output,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        parse_ok=parse_ok,
                        backend=generator.backend,
                        model_name=generator.model_name,
                        extra_fields={
                            "used_context_max_chars": used_context_chars,
                            "parse_mode": parse_mode,
                            # True when the response used the entire budget, i.e. it was almost
                            # certainly cut off rather than finishing on its own.
                            "hit_token_cap": completion_tokens >= args.query_max_new_tokens,
                        },
                    )
                    row = {
                        "fact_id": fact_id,
                        "hypothesis_idx": hyp_idx,
                        "candidate_tags": candidate_tags,
                        "verdicts_by_tag": verdicts_by_tag,
                        "parse_ok": parse_ok,
                        "parse_mode": parse_mode,
                        "call": call,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

                    if args.abort_check_after and generated == args.abort_check_after:
                        rate = (generated - parse_modes["failed"]) / generated
                        if rate < args.abort_if_parse_rate_below:
                            raise SystemExit(
                                f"\nABORTING after {generated} calls: only {rate:.0%} produced "
                                f"verdicts (threshold {args.abort_if_parse_rate_below:.0%}).\n"
                                f"parse modes: {dict(parse_modes)}\n"
                                f"Most likely cause is truncation -- check whether raw outputs "
                                f"end mid-JSON and raise --query-max-new-tokens (currently "
                                f"{args.query_max_new_tokens}). Inspect {raw_calls_path}.\n"
                                f"Re-run with --abort-check-after 0 to override."
                            )
                        print(
                            f"Parse-rate guard passed: {rate:.0%} of the first {generated} calls "
                            f"produced verdicts ({dict(parse_modes)}).",
                            flush=True,
                        )

                for tag in row["candidate_tags"]:
                    verdict = row["verdicts_by_tag"].get(tag, {})
                    for dimension in VERIFIER_DIMENSIONS:
                        opportunity_counts[dimension] += 1
                        if verdict.get(dimension) is not None:
                            firing_counts[dimension] += 1
                    all_verdicts.append(
                        {
                            "fact_id": fact_id,
                            "hypothesis_idx": hyp_idx,
                            "tag": tag,
                            "verdicts": {dimension: verdict.get(dimension) for dimension in VERIFIER_DIMENSIONS},
                        }
                    )

            if offset % args.log_every == 0:
                print(f"LLM verifier: {offset} facts processed", flush=True)
    finally:
        handle.close()
        if generator is not None:
            generator.close()

    firing_rate = {
        dimension: round(firing_counts[dimension] / opportunity_counts[dimension], 6) if opportunity_counts[dimension] else None
        for dimension in VERIFIER_DIMENSIONS
    }
    verdicts_path.write_text(json.dumps(all_verdicts, ensure_ascii=False) + "\n", encoding="utf-8")
    calls_with_verdicts = generated - parse_modes["failed"]
    parse_rate = round(calls_with_verdicts / generated, 6) if generated else None
    summary = {
        "verdicts_path": str(verdicts_path),
        "raw_calls_path": str(raw_calls_path),
        "top_m": args.top_m,
        "query_max_new_tokens": args.query_max_new_tokens,
        "firing_counts": dict(firing_counts),
        "opportunity_counts": dict(opportunity_counts),
        "firing_rate_per_dimension": firing_rate,
        # Read this BEFORE interpreting firing_rate. A parse failure and a genuine abstention
        # both drive the firing rate to zero, and they mean opposite things.
        "calls_generated": generated,
        "parse_modes": dict(parse_modes),
        "parse_rate": parse_rate,
        "resume_regenerated_unusable": unusable_existing,
        "note": (
            "A near-zero firing rate here (Appendix H found 5/1,160 FAMILY opportunities) "
            "means row 3.9 will show little difference from AGS full for a real reason -- "
            "abstention, not disagreement -- and that is the finding, not a bug. "
            "That reading is ONLY valid when parse_rate is high: a truncated or unparseable "
            "response also yields a zero firing rate, and that IS a bug. Check parse_rate and "
            "parse_modes first; if parse_modes['salvaged'] is large, responses are being cut "
            "off and --query-max-new-tokens needs raising."
        ),
    }
    (args.output_dir / "llm_verifier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
