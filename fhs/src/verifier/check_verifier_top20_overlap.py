#!/usr/bin/env python3
"""Does the LLM verification layer (row 3.9) change what the listwise reranker actually sees?

Table 5 scores every row at the retrieval stage, with no listwise reranker. But the deployed
pipeline DOES rerank: run_fintagging_grounding_baseline.py feeds the top --rerank-list-size
(default 20) of the retrieval ranking to Qwen3-32B and lets it reorder them. The verifier is
the same model doing a similar job on an overlapping candidate window (M=10 cluster
representatives), so row 3.9's pre-reranker gain (+0.076 Acc.) may not survive end to end.

This script answers the decisive question WITHOUT a GPU: does the verifier change the top-20
*set* handed to the reranker, or only its internal order?

  - set unchanged on ~all facts  -> the reranker receives the same material either way, so the
    verifier cannot change the deployed answer except through the reranker's own position
    bias. Row 3.9's gain would be an artifact of measuring before a stage that overwrites it.
  - set changes on a meaningful fraction -> the gain is real but partial, and the honest number
    is an end-to-end rerun (GPU), not this table.

Usage:
  conda activate finben
  cd /nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/fhs
  python verifier/check_verifier_top20_overlap.py                # full, ~35 min CPU
  python verifier/check_verifier_top20_overlap.py --limit 50     # ~1 min smoke test
"""

from __future__ import annotations
# --- resolve local packages regardless of this file's depth in the tree ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "src" / "run_fintagging_grounding_baseline.py").exists():
        _sys.path.insert(0, str(_p / "src"))
        _sys.path.insert(0, str(_p / "analysis"))
        FHS_ROOT = _p
        break
# -------------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import AblationConfig, evaluate, normalize_tags, reset_consensus_cache  # noqa: E402
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts  # noqa: E402
from verifier.run_test_rows import load_llm_verifier_verdicts  # noqa: E402

DEFAULT_VERDICTS = (
    _PARENT / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_verdicts.json"
)
DEFAULT_OUT = _PARENT / "runs_ags_table5_ablation" / "qwen3_32b" / "verifier_top20_overlap.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_TEST_TRACE)
    parser.add_argument("--llm-verifier-verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--rerank-list-size", type=int, default=20,
                        help="Must match --rerank-list-size of the deployed run (default 20).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def gold_rank(tags: list[str], gold: set[str]) -> int | None:
    for index, tag in enumerate(tags, start=1):
        if tag in gold:
            return index
    return None


def main() -> None:
    args = parse_args()
    n_top = args.rerank_list_size

    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    verdicts = load_llm_verifier_verdicts(args.llm_verifier_verdicts)
    if not verdicts:
        raise SystemExit(f"No verdicts loaded from {args.llm_verifier_verdicts}")
    print(f"verdicts loaded: {len(verdicts)}", flush=True)

    facts = list(load_test_facts(args.test_trace, limit=args.limit).values())
    print(f"test facts: {len(facts)}", flush=True)

    def run(name: str, config: AblationConfig) -> list[dict]:
        reset_consensus_cache()
        return [evaluate(fact, config, normalization_map) for fact in tqdm(facts, desc=name, unit="fact")]

    full_rows = run("AGS (full)", AblationConfig(name="AGS (full)", beta=0.6))
    verifier_rows = run(
        "llm_verifier",
        AblationConfig(name="+ LLM verification layer", beta=0.6, llm_verifier_verdicts=verdicts),
    )

    n = len(facts)
    set_same = order_same = 0
    gold_enters = gold_leaves = gold_moves = 0
    full_hits = verif_hits = 0
    changed_examples: list[dict] = []

    for fact, frow, vrow in zip(facts, full_rows, verifier_rows):
        ftop = frow["candidate_tags"][:n_top]
        vtop = vrow["candidate_tags"][:n_top]
        gold = set(normalize_tags(fact.gold_tags))

        if set(ftop) == set(vtop):
            set_same += 1
        elif len(changed_examples) < 15:
            changed_examples.append(
                {
                    "fact_id": frow["fact_id"],
                    "modality": frow["modality"],
                    "dropped_from_top": sorted(set(ftop) - set(vtop))[:5],
                    "added_to_top": sorted(set(vtop) - set(ftop))[:5],
                }
            )
        if ftop == vtop:
            order_same += 1

        frank, vrank = gold_rank(ftop, gold), gold_rank(vtop, gold)
        full_hits += frank is not None
        verif_hits += vrank is not None
        if frank is None and vrank is not None:
            gold_enters += 1
        elif frank is not None and vrank is None:
            gold_leaves += 1
        elif frank is not None and vrank is not None and frank != vrank:
            gold_moves += 1

    pct = lambda x: round(100.0 * x / n, 3) if n else None
    result = {
        "rerank_list_size": n_top,
        "n_facts": n,
        "top_n_set_identical": set_same,
        "top_n_set_identical_pct": pct(set_same),
        "top_n_order_identical": order_same,
        "top_n_order_identical_pct": pct(order_same),
        f"recall_at_{n_top}_ags_full": pct(full_hits),
        f"recall_at_{n_top}_verifier": pct(verif_hits),
        "gold_enters_top_n": gold_enters,
        "gold_leaves_top_n": gold_leaves,
        "gold_changes_rank_within_top_n": gold_moves,
        "changed_examples": changed_examples,
        "interpretation": (
            "If top_n_set_identical_pct is ~100 and recall_at_N is unchanged, the listwise "
            "reranker receives the same candidates under both configurations, so row 3.9 "
            "cannot improve the deployed end-to-end accuracy -- its Table 5 gain is measured "
            "before a stage that re-sorts the same items anyway. Note that even with an "
            "identical SET, a different presentation ORDER can still perturb a listwise "
            "reranker through position bias; that is noise, not signal, and does not rescue "
            "the row."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\n=== verifier vs AGS full, top-%d handed to the reranker ===" % n_top, flush=True)
    for key in (
        "n_facts",
        "top_n_set_identical_pct",
        "top_n_order_identical_pct",
        f"recall_at_{n_top}_ags_full",
        f"recall_at_{n_top}_verifier",
        "gold_enters_top_n",
        "gold_leaves_top_n",
        "gold_changes_rank_within_top_n",
    ):
        print(f"  {key:<34} {result[key]}", flush=True)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
