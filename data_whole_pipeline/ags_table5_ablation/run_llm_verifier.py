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
    parser.add_argument("--query-max-new-tokens", type=int, default=512)
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


def parse_verifier_output(raw_output: str, candidate_tags: list[str]) -> tuple[dict[str, dict[str, Any]], bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    verdicts_by_tag: dict[str, dict[str, Any]] = {}
    entries = parsed.get("verdicts")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tag = normalize_tag(entry.get("tag", ""))
            if tag not in candidate_tags:
                continue
            verdict = {}
            for dimension in VERIFIER_DIMENSIONS:
                raw_verdict = str(entry.get(dimension, "")).strip().lower()
                verdict[dimension] = VERDICT_TO_MATCHED.get(raw_verdict, None)
            verdict["confidence"] = float(entry.get("confidence", 0.0) or 0.0)
            verdicts_by_tag[tag] = verdict
    return verdicts_by_tag, bool(parse_ok and verdicts_by_tag)


def load_existing(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    existing: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return existing
    for row in stream_jsonl(path):
        existing[(int(row["fact_id"]), int(row["hypothesis_idx"]))] = row
    return existing


def main() -> None:
    args = parse_args()
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_calls_path = args.output_dir / "llm_verifier_calls.jsonl"
    verdicts_path = args.output_dir / "llm_verifier_verdicts.json"

    existing = load_existing(raw_calls_path) if args.resume else {}
    firing_counts = Counter()
    opportunity_counts = Counter()
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
                    verdicts_by_tag, parse_ok = parse_verifier_output(raw_output, candidate_tags)
                    call = llm_call_record(
                        "table5_llm_verifier",
                        raw_output=raw_output,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=generator.count_text_tokens(raw_output),
                        parse_ok=parse_ok,
                        backend=generator.backend,
                        model_name=generator.model_name,
                        extra_fields={"used_context_max_chars": used_context_chars},
                    )
                    row = {
                        "fact_id": fact_id,
                        "hypothesis_idx": hyp_idx,
                        "candidate_tags": candidate_tags,
                        "verdicts_by_tag": verdicts_by_tag,
                        "parse_ok": parse_ok,
                        "call": call,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

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
    summary = {
        "verdicts_path": str(verdicts_path),
        "raw_calls_path": str(raw_calls_path),
        "top_m": args.top_m,
        "firing_counts": dict(firing_counts),
        "opportunity_counts": dict(opportunity_counts),
        "firing_rate_per_dimension": firing_rate,
        "note": (
            "A near-zero firing rate here (Appendix H found 5/1,160 FAMILY opportunities) "
            "means row 3.9 will show little difference from AGS full for a real reason -- "
            "abstention, not disagreement -- and that is the finding, not a bug."
        ),
    }
    (args.output_dir / "llm_verifier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
