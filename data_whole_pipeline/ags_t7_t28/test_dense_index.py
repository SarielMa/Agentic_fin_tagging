"""Offline tests for the T28 dense/hybrid retrievers (no GPU, no taxonomy load).

Targeted check of DenseRetriever/HybridRetriever on a tiny synthetic taxonomy.

Exercises the interface contract that lets these drop into retrieve_candidates() unchanged,
plus the coverage-on pool path (which rescores the whole type subset) and hybrid RRF.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from run_fintagging_grounding_baseline import Concept, retrieve_candidates
from ags_t7_t28.dense_index import DenseRetriever, HybridRetriever, build_retriever

MODEL = "BAAI/bge-small-en-v1.5"


def make_concepts():
    raw = [
        ("Assets", "monetaryItemType", "Assets", "Sum of the carrying amounts of all assets."),
        ("Liabilities", "monetaryItemType", "Liabilities", "Sum of the carrying amounts of all liabilities."),
        ("Revenues", "monetaryItemType", "Revenues", "Amount of revenue recognized from goods sold or services rendered."),
        ("Goodwill", "monetaryItemType", "Goodwill", "Amount after accumulated impairment loss of an asset from acquisition."),
        ("DeferredTaxAssetsNet", "monetaryItemType", "Deferred Tax Assets Net", "Amount after allocation of valuation allowances of deferred tax asset."),
        ("SharesOutstanding", "sharesItemType", "Shares Outstanding", "Number of shares issued and outstanding."),
        ("EmployeeCount", "integerItemType", "Employee Count", "Number of employees."),
    ]
    return [
        Concept(
            tag=f"us-gaap:{tag}",
            raw_tag=tag,
            entity_type=etype,
            standard_label=label,
            documentation=doc,
            references=[],
            retrieval_text=f"{tag}. {label}. {doc}",
        )
        for tag, etype, label, doc in raw
    ]


def main():
    concepts = make_concepts()
    failures = []

    def check(name, condition, detail=""):
        print(f"{'PASS' if condition else 'FAIL'}  {name} {detail}")
        if not condition:
            failures.append(name)

    dense = DenseRetriever(concepts, model_name=MODEL, device="cpu", show_progress=False)

    # 1. interface contract: retrieve_candidates() works unchanged on a dense retriever
    cands = retrieve_candidates(dense, "total assets of the company", "monetaryItemType", 3)
    expected_keys = {
        "rank", "tag", "type", "standard_label", "documentation", "references",
        "retrieval_text", "bm25_score", "bm25_normalized_score", "label_coverage",
        "query_label_coverage", "retrieval_score",
    }
    check("dense retrieve_candidates returns full candidate dicts", set(cands[0]) == expected_keys)
    check("dense honors top_k", len(cands) == 3, f"got {len(cands)}")
    check("dense ranks are 1..k", [c["rank"] for c in cands] == [1, 2, 3])
    check("dense type filter respected", all(c["type"] == "monetaryItemType" for c in cands))
    check("dense finds Assets for an assets query", cands[0]["tag"] == "us-gaap:Assets", cands[0]["tag"])

    # 2. type filter actually restricts (sharesItemType has exactly one member)
    shares = retrieve_candidates(dense, "number of shares", "sharesItemType", 5)
    check("dense type subset limits results", len(shares) == 1 and shares[0]["tag"] == "us-gaap:SharesOutstanding")

    # 3. coverage-on path: rescores the whole type subset, not a truncated pool
    dense.label_coverage_weight = 1.0
    cov = retrieve_candidates(dense, "goodwill", "monetaryItemType", 3)
    check("dense +cov returns top_k", len(cov) == 3, f"got {len(cov)}")
    check("dense +cov populates coverage fields", any(c["label_coverage"] > 0 for c in cov))
    check("dense +cov surfaces the exact label", cov[0]["tag"] == "us-gaap:Goodwill", cov[0]["tag"])
    check(
        "dense +cov retrieval_score = normalized + w*(cov+qcov)",
        abs(cov[0]["retrieval_score"] - (cov[0]["bm25_normalized_score"] + cov[0]["label_coverage"] + cov[0]["query_label_coverage"])) < 1e-6,
    )
    dense.label_coverage_weight = 0.0

    # 4. hybrid: shares the dense index, fuses with BM25 by RRF
    bm25 = build_retriever("bm25", concepts, label_coverage_weight=0.0)
    hybrid = HybridRetriever(concepts, dense, label_coverage_weight=0.0, bm25_retriever=bm25)
    hyb = retrieve_candidates(hybrid, "total assets of the company", "monetaryItemType", 3)
    check("hybrid returns full candidate dicts", set(hyb[0]) == expected_keys)
    check("hybrid ranks are 1..k", [c["rank"] for c in hyb] == [1, 2, 3])
    check("hybrid finds Assets", hyb[0]["tag"] == "us-gaap:Assets", hyb[0]["tag"])

    # 5. the leak guard: a shared BM25/dense instance left at w_cov=1.0 by an earlier grid row
    #    must not change hybrid's own (coverage-off) output.
    baseline = [c["tag"] for c in retrieve_candidates(hybrid, "goodwill impairment", "monetaryItemType", 4)]
    bm25.label_coverage_weight = 1.0
    dense.label_coverage_weight = 1.0
    after = [c["tag"] for c in retrieve_candidates(hybrid, "goodwill impairment", "monetaryItemType", 4)]
    check("hybrid ignores leaked coverage weights on shared sides", baseline == after, f"{baseline} vs {after}")
    check("hybrid restores the sides' weights", bm25.label_coverage_weight == 1.0 and dense.label_coverage_weight == 1.0)
    bm25.label_coverage_weight = 0.0
    dense.label_coverage_weight = 0.0

    # 6. build_retriever reuse: same object back, weight re-pointed
    again = build_retriever("dense", concepts, label_coverage_weight=1.0, dense=dense)
    check("build_retriever reuses the dense index", again is dense and dense.label_coverage_weight == 1.0)
    dense.label_coverage_weight = 0.0
    again_bm25 = build_retriever("bm25", concepts, label_coverage_weight=1.0, bm25=bm25)
    check("build_retriever reuses the bm25 index", again_bm25 is bm25 and bm25.label_coverage_weight == 1.0)
    bm25.label_coverage_weight = 0.0

    # 7. query cache priming produces identical results to on-demand encoding
    fresh = DenseRetriever(concepts, model_name=MODEL, device="cpu", show_progress=False)
    fresh.prime_query_cache(["total assets of the company"], show_progress=False)
    primed = [c["tag"] for c in retrieve_candidates(fresh, "total assets of the company", "monetaryItemType", 3)]
    ondemand = [c["tag"] for c in retrieve_candidates(dense, "total assets of the company", "monetaryItemType", 3)]
    check("primed cache matches on-demand encoding", primed == ondemand, f"{primed} vs {ondemand}")

    # 8. embedding cache round-trip: the CPU replay stage depends on a cache-loaded retriever
    #    being indistinguishable from a freshly embedded one, and on it never touching the
    #    sentence-transformer at all (stage B has no GPU and should not pay to load one).
    import tempfile

    queries = ["total assets of the company", "goodwill impairment", "number of shares"]
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = Path(tmp) / "dense_embeddings.pt"
        writer = DenseRetriever(concepts, model_name=MODEL, device="cpu", show_progress=False,
                                cache_path=cache_file)
        writer.prime_query_cache(queries, show_progress=False)
        writer.save_cache()
        check("save_cache writes the cache file", cache_file.exists())

        reader = DenseRetriever(concepts, model_name=MODEL, device="cpu", show_progress=False,
                                cache_path=cache_file)
        check("cache-loaded retriever never loads the model", reader._model is None)
        check(
            "cache round-trips the concept embeddings bit-exactly",
            bool(reader.embeddings.equal(writer.embeddings)),
        )
        for query in queries:
            hit_a = [c["tag"] for c in retrieve_candidates(writer, query, "monetaryItemType", 4)]
            hit_b = [c["tag"] for c in retrieve_candidates(reader, query, "monetaryItemType", 4)]
            check(f"cached retrieval identical for {query!r}", hit_a == hit_b, f"{hit_a} vs {hit_b}")
        reader.label_coverage_weight = 1.0
        writer.label_coverage_weight = 1.0
        cov_a = [c["retrieval_score"] for c in retrieve_candidates(writer, "goodwill", "monetaryItemType", 3)]
        cov_b = [c["retrieval_score"] for c in retrieve_candidates(reader, "goodwill", "monetaryItemType", 3)]
        check("cached +cov scores identical", cov_a == cov_b, f"{cov_a} vs {cov_b}")
        reader.label_coverage_weight = 0.0
        writer.label_coverage_weight = 0.0

        # A cache built for a different model or a different taxonomy must be refused, not
        # silently reused -- that would produce plausible, wrong numbers.
        other = DenseRetriever(concepts[:-1], model_name=MODEL, device="cpu",
                               show_progress=False, cache_path=cache_file)
        check("cache rejected when the taxonomy differs", other._model is not None)
        check("rejected cache still yields a working index", other.embeddings.shape[0] == len(concepts) - 1)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("all retriever checks passed")


if __name__ == "__main__":
    main()
