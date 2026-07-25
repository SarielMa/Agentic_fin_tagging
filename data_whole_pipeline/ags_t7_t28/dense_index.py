#!/usr/bin/env python3
"""T28 dense and hybrid retrievers (comparing_methods/ags_t7_t28_spec.md, "T28 -- Retriever
Robustness", step 1).

Both index the identical 17,388-concept taxonomy with the identical text representation BM25
uses -- `Concept.retrieval_text`, i.e. "{tag}. {standard_label}. {documentation}" as built by
build_us_gaap_2024_retrieval_dataset.build_retrieval_text and tokenized by
run_fintagging_grounding_baseline.BM25Index (:452-459). Only the retrieval model changes.

Every retriever here exposes the same `retrieve(query, entity_type, top_k)` contract as
`TaxonomyRetriever` -- a list of
`(concept, score, label_coverage, query_label_coverage, normalized_score, retrieval_score)`
6-tuples -- so `run_fintagging_grounding_baseline.retrieve_candidates` and everything
downstream of it (fusion, `agree()`, the rerank, `ags_table5_ablation.core.evaluate`) run
against a dense or hybrid index completely unchanged. That is what lets T28 re-run
consolidation "exactly as in the frozen AGS config" instead of reimplementing it.

Note on field names: `concept_to_candidate` writes the score slots to keys literally named
`bm25_score` / `bm25_normalized_score`. Under these retrievers those keys carry dense cosine
or hybrid RRF scores. Nothing downstream reads them for ranking -- `core.fuse` keys off `rank`
alone -- but the metrics.json written by the T28 runner records which retriever produced them
so a stored trace can never be misread as BM25.

The coverage term (Eq. 10) is a lexical feature and its formula is reused verbatim from
`TaxonomyRetriever.label_coverage` / `.query_label_coverage` so the "+cov" rows differ from
the plain rows only in the retriever underneath, not in how coverage is computed. Per the
spec, dense and hybrid rows are reported WITHOUT it by default; "+cov" is an explicitly marked
extra variant for AGS only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from run_fintagging_grounding_baseline import (  # noqa: E402
    BM25Index,
    Concept,
    TaxonomyRetriever,
    normalize_tag,
    tokenize,
)

DEFAULT_DENSE_MODEL = "BAAI/bge-large-en-v1.5"
# BGE retrieval models expect this instruction on the query side only; passages stay bare.
# E5-class models use "query: " / "passage: " instead -- set --query-instruction accordingly.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_RRF_KAPPA = 60.0  # matches the frozen AGS fusion kappa


def default_query_instruction(model_name: str) -> str:
    lowered = model_name.lower()
    if "bge" in lowered and "zh" not in lowered:
        return BGE_QUERY_INSTRUCTION
    if "e5" in lowered:
        return "query: "
    return ""


def default_passage_prefix(model_name: str) -> str:
    return "passage: " if "e5" in model_name.lower() else ""


class _CoverageMixin:
    """The Eq. 10 lexical coverage term, delegated to TaxonomyRetriever's own implementation so
    a "+cov" row cannot drift from how the BM25 rows compute it."""

    label_coverage_weight: float = 0.0

    @staticmethod
    def label_coverage(query_tokens: set[str], concept: Concept) -> float:
        return TaxonomyRetriever.label_coverage(None, query_tokens, concept)  # type: ignore[arg-type]

    @staticmethod
    def query_label_coverage(query_tokens: set[str], concept: Concept) -> float:
        return TaxonomyRetriever.query_label_coverage(None, query_tokens, concept)  # type: ignore[arg-type]

    def _apply_coverage(
        self,
        scored: list[tuple[Concept, float, float]],
        query: str,
        top_k: int,
    ) -> list[tuple[Concept, float, float, float, float, float]]:
        """scored: (concept, raw_score, normalized_score) already sorted best-first.

        Mirrors TaxonomyRetriever.retrieve (:536-565) exactly: with the term off, the
        normalized score IS the retrieval score and the order is untouched; with it on, the
        rescore is `normalized + w_cov * (coverage + query_coverage)` and the same tie-break
        chain applies.
        """
        if self.label_coverage_weight <= 0.0:
            return [
                (concept, raw, 0.0, 0.0, normalized, normalized)
                for concept, raw, normalized in scored[:top_k]
            ]

        query_tokens = set(tokenize(query))
        rescored: list[tuple[Concept, float, float, float, float, float]] = []
        for concept, raw, normalized in scored:
            coverage = self.label_coverage(query_tokens, concept)
            query_coverage = self.query_label_coverage(query_tokens, concept)
            retrieval_score = normalized + self.label_coverage_weight * (coverage + query_coverage)
            rescored.append((concept, raw, coverage, query_coverage, normalized, retrieval_score))
        rescored.sort(
            key=lambda item: (-item[5], -item[2], -item[3], -item[1], normalize_tag(item[0].tag))
        )
        return rescored[:top_k]


def _range_normalize(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [1.0] * len(scores)
    return [(score - lo) / (hi - lo) for score in scores]


class DenseRetriever(_CoverageMixin):
    """Exact cosine nearest-neighbour over the taxonomy's sentence embeddings.

    The spec calls for exact search on purpose: "17k is small -- exact cosine is fine and
    removes an approximation confound", so there is no HNSW/faiss approximation here and no
    faiss dependency.

    Type filtering mirrors `TaxonomyRetriever(type_filter=True)`: candidates are restricted to
    the query's own entity type when that type exists in the taxonomy, otherwise the full set
    is searched. The restriction is a mask over one shared embedding matrix rather than a
    separate index per type -- embeddings do not change with the subset, unlike BM25's IDF.
    """

    def __init__(
        self,
        concepts: list[Concept],
        model_name: str = DEFAULT_DENSE_MODEL,
        type_filter: bool = True,
        label_coverage_weight: float = 0.0,
        batch_size: int = 256,
        device: str | None = None,
        query_instruction: str | None = None,
        passage_prefix: str | None = None,
        show_progress: bool = True,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.concepts = concepts
        self.model_name = model_name
        self.type_filter = type_filter
        self.label_coverage_weight = label_coverage_weight
        self.query_instruction = (
            default_query_instruction(model_name) if query_instruction is None else query_instruction
        )
        self.passage_prefix = (
            default_passage_prefix(model_name) if passage_prefix is None else passage_prefix
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

        # The identical text representation BM25 indexes (module docstring).
        passages = [f"{self.passage_prefix}{concept.retrieval_text}" for concept in concepts]
        print(
            f"Embedding {len(passages)} concepts with {model_name} on {self.device} ...",
            flush=True,
        )
        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,  # cosine == dot product once both sides are unit norm
            show_progress_bar=show_progress,
        )
        self.embeddings = embeddings.to(self.device)

        self.type_indices: dict[str, Any] = {}
        for index, concept in enumerate(concepts):
            self.type_indices.setdefault(concept.entity_type, []).append(index)
        self.type_indices = {
            entity_type: torch.tensor(indices, dtype=torch.long, device=self.device)
            for entity_type, indices in self.type_indices.items()
        }
        self._query_cache: dict[str, Any] = {}

    def prime_query_cache(self, queries: Sequence[str], show_progress: bool = True) -> None:
        """Batch-encode every query the run will issue.

        T28 replays thousands of logged query strings; encoding them one at a time through
        `retrieve()` would dominate the runtime. Queries are reused verbatim (no regeneration),
        so they are all known up front.
        """
        pending = sorted({query for query in queries if query and query not in self._query_cache})
        if not pending:
            return
        print(f"Embedding {len(pending)} unique queries ...", flush=True)
        vectors = self.model.encode(
            [f"{self.query_instruction}{query}" for query in pending],
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).to(self.device)
        for offset, query in enumerate(pending):
            self._query_cache[query] = vectors[offset]

    def encode_query(self, query: str) -> Any:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        vector = self.model.encode(
            f"{self.query_instruction}{query}",
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).to(self.device)
        self._query_cache[query] = vector
        return vector

    def retrieve(
        self, query: str, entity_type: str, top_k: int
    ) -> list[tuple[Concept, float, float, float, float, float]]:
        torch = self._torch
        subset = self.type_indices.get(entity_type) if self.type_filter else None
        if subset is None:
            matrix = self.embeddings
            row_ids = None
        else:
            matrix = self.embeddings.index_select(0, subset)
            row_ids = subset

        # The coverage term rescores a pool, so it needs more than top_k to work with. The
        # frozen AGS retriever runs with label_coverage_pool_multiplier=0, i.e. it rescores the
        # WHOLE type-filtered subset; matching that here keeps a "+cov" row comparable across
        # retrievers instead of letting dense rescore a narrower pool than BM25 did.
        pool_k = matrix.shape[0] if self.label_coverage_weight > 0 else min(matrix.shape[0], top_k)
        with torch.no_grad():
            scores = matrix @ self.encode_query(query)
            top = torch.topk(scores, k=pool_k)
        values = top.values.tolist()
        positions = top.indices.tolist()
        if row_ids is not None:
            positions = row_ids[top.indices].tolist()

        normalized = _range_normalize(values)
        scored = [
            (self.concepts[index], float(score), float(norm))
            for index, score, norm in zip(positions, values, normalized)
        ]
        return self._apply_coverage(scored, query, top_k)


class HybridRetriever(_CoverageMixin):
    """BM25 + dense, combined with RRF.

    The spec prefers RRF over a score-normalized linear blend ("RRF is simpler and
    parameter-light, so prefer it and state kappa"); kappa is recorded in the runner's
    metrics.json. Both sides contribute a ranked list of `fusion_pool_k` candidates and the
    fused score is `sum_r 1/(kappa + rank_r)` -- the same functional form as the frozen AGS
    fusion, so a reader comparing the two sees one fusion rule, not two.

    One asymmetry worth stating in the caption if a "+cov" hybrid row is reported: coverage is
    applied after fusion, so it can only reorder concepts one of the two sides already
    surfaced within `fusion_pool_k`. The BM25 and dense "+cov" rows rescore their whole
    type-filtered subset, which hybrid structurally cannot.
    """

    def __init__(
        self,
        concepts: list[Concept],
        dense: DenseRetriever,
        type_filter: bool = True,
        label_coverage_weight: float = 0.0,
        rrf_kappa: float = DEFAULT_RRF_KAPPA,
        fusion_pool_k: int = 200,
        bm25_retriever: TaxonomyRetriever | None = None,
    ) -> None:
        self.concepts = concepts
        self.dense = dense
        self.type_filter = type_filter
        self.label_coverage_weight = label_coverage_weight
        self.rrf_kappa = rrf_kappa
        self.fusion_pool_k = fusion_pool_k
        # The BM25 side runs with its own coverage term OFF: the hybrid row's coverage setting
        # is applied once, after fusion, so "+cov" means the same thing for every retriever.
        # `retrieve` forces this to 0.0 per call rather than trusting it here, because a shared
        # TaxonomyRetriever is also handed to the BM25 rows, which do set it to 1.0.
        self.bm25 = bm25_retriever or TaxonomyRetriever(
            concepts,
            type_filter=type_filter,
            label_coverage_weight=0.0,
            label_coverage_pool_multiplier=0,
        )
        self.by_tag = {concept.tag: concept for concept in concepts}

    def prime_query_cache(self, queries: Sequence[str], show_progress: bool = True) -> None:
        self.dense.prime_query_cache(queries, show_progress=show_progress)

    def retrieve(
        self, query: str, entity_type: str, top_k: int
    ) -> list[tuple[Concept, float, float, float, float, float]]:
        pool_k = max(top_k, self.fusion_pool_k)
        # Coverage is applied once, post-fusion, so BOTH sides are fetched with it off. Both
        # retrievers may be shared with the BM25/dense grid rows, which do set the weight to
        # 1.0, so neither side's current setting can be trusted -- zero and restore each.
        dense_weight = self.dense.label_coverage_weight
        bm25_weight = self.bm25.label_coverage_weight
        self.dense.label_coverage_weight = 0.0
        self.bm25.label_coverage_weight = 0.0
        try:
            dense_hits = self.dense.retrieve(query, entity_type, pool_k)
            bm25_hits = self.bm25.retrieve(query, entity_type, pool_k)
        finally:
            self.dense.label_coverage_weight = dense_weight
            self.bm25.label_coverage_weight = bm25_weight

        fused: dict[str, float] = {}
        concepts: dict[str, Concept] = {}
        for hits in (bm25_hits, dense_hits):
            for rank, item in enumerate(hits, start=1):
                concept = item[0]
                tag = concept.tag
                concepts[tag] = concept
                fused[tag] = fused.get(tag, 0.0) + 1.0 / (self.rrf_kappa + rank)

        order = sorted(fused, key=lambda tag: (-fused[tag], normalize_tag(tag)))
        raw_scores = [fused[tag] for tag in order]
        normalized = _range_normalize(raw_scores)
        scored = [
            (concepts[tag], float(raw), float(norm))
            for tag, raw, norm in zip(order, raw_scores, normalized)
        ]
        return self._apply_coverage(scored, query, top_k)


def build_retriever(
    kind: str,
    concepts: list[Concept],
    model_name: str = DEFAULT_DENSE_MODEL,
    label_coverage_weight: float = 0.0,
    type_filter: bool = True,
    rrf_kappa: float = DEFAULT_RRF_KAPPA,
    dense: DenseRetriever | None = None,
    bm25: TaxonomyRetriever | None = None,
    batch_size: int = 256,
    device: str | None = None,
    query_instruction: str | None = None,
) -> Any:
    """`kind` in {"bm25", "dense", "hybrid"}.

    `dense` and `bm25` let callers build each index once and reuse it across grid rows --
    embedding 17k concepts and tokenizing 17k BM25 documents are both far more expensive than
    the retrieval they enable. The coverage weight is a plain attribute read at query time
    (TaxonomyRetriever.retrieve:536), so re-pointing it per row is safe and is what the
    pipeline itself does (run_fintagging_grounding_baseline.py:3391).
    """
    if kind == "bm25":
        if bm25 is not None:
            bm25.label_coverage_weight = label_coverage_weight
            return bm25
        return TaxonomyRetriever(
            concepts,
            type_filter=type_filter,
            label_coverage_weight=label_coverage_weight,
            label_coverage_pool_multiplier=0,
        )
    if kind == "dense":
        if dense is not None:
            dense.label_coverage_weight = label_coverage_weight
            return dense
        return DenseRetriever(
            concepts,
            model_name=model_name,
            type_filter=type_filter,
            label_coverage_weight=label_coverage_weight,
            batch_size=batch_size,
            device=device,
            query_instruction=query_instruction,
        )
    if kind == "hybrid":
        base = dense or DenseRetriever(
            concepts,
            model_name=model_name,
            type_filter=type_filter,
            label_coverage_weight=0.0,
            batch_size=batch_size,
            device=device,
            query_instruction=query_instruction,
        )
        return HybridRetriever(
            concepts,
            base,
            type_filter=type_filter,
            label_coverage_weight=label_coverage_weight,
            rrf_kappa=rrf_kappa,
            bm25_retriever=bm25,
        )
    raise ValueError(f"Unknown retriever kind {kind!r}; expected 'bm25', 'dense' or 'hybrid'")


__all__ = [
    "BM25Index",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_RRF_KAPPA",
    "DenseRetriever",
    "HybridRetriever",
    "build_retriever",
]
