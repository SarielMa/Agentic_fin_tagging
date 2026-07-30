#!/usr/bin/env python3
"""Gate a batch submission on the CONFIGURATION a smoke recorded, not on whether it finished.

Twice now a smoke was read as success while the run was broken: one passed its startup assertions
while every verifier response was truncated (0/171 clean parses), and one reported no truncation
because REUSE_CANDIDATES=1 meant it never generated anything at all. Both would have been caught by
asserting the settings and the fact that work happened.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

def main() -> int:
    d = Path(sys.argv[1])
    metrics = next((d / n for n in ("metrics.json", "bm25_metrics.json") if (d / n).exists()), None)
    if metrics is None:
        print(f"FAIL  no metrics file in {d}")
        return 1
    j = json.loads(metrics.read_text(encoding="utf-8"))
    t = j.get("truncation") or {}
    checks = {
        "generation actually happened (calls > 0)": (t.get("generation_calls") or 0) > 0,
        "shared generation cap == 2048": t.get("token_cap") == 2048,
        "no call reached the cap": t.get("calls_at_token_cap") == 0,
        "label coverage weight == 1.0": j.get("label_coverage_weight") == 1.0,
        "top_k == 200": j.get("top_k") == 200,
        "rrf kappa == 60": float(j.get("rrf_kappa") or 0) == 60.0,
    }
    rate = j.get("query_generation_parse_success_rate")
    if rate is not None:
        checks["generation parse rate == 1.0"] = rate == 1.0
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"  recorded truncation block: {t}")
    bad = [n for n, ok in checks.items() if not ok]
    print("GATE OPEN" if not bad else f"GATE CLOSED: {len(bad)} failing")
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main())
