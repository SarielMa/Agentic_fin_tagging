from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

from .data import TaxonomyConcept, tag_terms
from .memory_store import LTMStore
from .text_utils import localize_context, normalize_space, rewrite_evidence_for_retrieval


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TAXONOMY_DOC_MODES = ("text", "text_tag_terms", "full")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))
    if max_score <= 0 and min_score >= 0:
        return np.zeros_like(scores, dtype=float)
    if math.isclose(max_score, min_score):
        return np.ones_like(scores, dtype=float) if max_score > 0 else np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def taxonomy_document(concept: TaxonomyConcept, mode: str = "full") -> str:
    if mode == "text":
        return normalize_space(concept.text)
    if mode == "text_tag_terms":
        tag_text = tag_terms(concept.tag)
        if normalize_space(tag_text).lower() == normalize_space(concept.text).lower():
            return normalize_space(concept.text)
        return normalize_space(f"{concept.text} {tag_text}")
    if mode == "full":
        return normalize_space(f"{concept.text} {tag_terms(concept.tag)} {concept.tag} {concept.entity_type}")
    raise ValueError(f"Unsupported taxonomy_doc_mode: {mode}. Choose one of {', '.join(TAXONOMY_DOC_MODES)}.")


class BM25Index:
    """Small dependency-free BM25 index."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in docs]
        self.doc_lengths = np.array([len(tokens) for tokens in self.doc_tokens], dtype=float)
        self.avg_doc_length = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        doc_freq: Counter[str] = Counter()
        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            doc_freq.update(counts.keys())
            for token, freq in counts.items():
                self.postings.setdefault(token, []).append((doc_idx, freq))
        n_docs = len(docs)
        self.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.doc_tokens), dtype=float)
        if len(self.doc_tokens) == 0 or self.avg_doc_length <= 0:
            return scores
        for token in set(tokenize(query)):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * freq * (self.k1 + 1.0) / denom
        return scores


class DenseTextIndex:
    """Dense text index with optional sentence-transformer and local SVD fallback."""

    def __init__(self, docs: list[str], model_name: str = "", n_components: int = 128) -> None:
        self.docs = docs
        self.model_name = model_name
        self.backend = "svd"
        self.model: Any | None = None
        self.vectorizer: HashingVectorizer | None = None
        self.reducer: TruncatedSVD | None = None
        self.matrix = self._build_matrix(docs, model_name, n_components)

    def _build_matrix(self, docs: list[str], model_name: str, n_components: int) -> np.ndarray:
        if not docs:
            return np.zeros((0, 0), dtype=float)

        if model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
                self.backend = "sentence_transformers"
                encoded = self.model.encode(
                    docs,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return np.asarray(encoded, dtype=float)
            except Exception:
                self.model = None
                self.backend = "svd"

        self.vectorizer = HashingVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            n_features=2**16,
            alternate_sign=False,
            norm=None,
        )
        hashed = self.vectorizer.transform(docs)
        max_components = min(hashed.shape[0] - 1, hashed.shape[1] - 1, n_components)
        if max_components >= 2 and hashed.nnz:
            self.reducer = TruncatedSVD(n_components=max_components, random_state=0)
            dense = self.reducer.fit_transform(hashed)
        else:
            dense = hashed.toarray()
        return sklearn_normalize(np.asarray(dense, dtype=float), norm="l2", copy=False)

    def scores(self, query: str) -> np.ndarray:
        if self.matrix.shape[0] == 0:
            return np.zeros(0, dtype=float)
        if self.backend == "sentence_transformers" and self.model is not None:
            encoded = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vec = np.asarray(encoded, dtype=float)
        else:
            if self.vectorizer is None:
                return np.zeros(self.matrix.shape[0], dtype=float)
            hashed = self.vectorizer.transform([query])
            if self.reducer is not None:
                query_vec = self.reducer.transform(hashed)
            else:
                query_vec = hashed.toarray()
            query_vec = sklearn_normalize(np.asarray(query_vec, dtype=float), norm="l2", copy=False)
        return np.asarray(query_vec @ self.matrix.T).ravel()


class HybridTextIndex:
    """BM25 plus dense retrieval over a fixed list of documents."""

    def __init__(
        self,
        docs: list[str],
        bm25_weight: float = 1.0,
        dense_weight: float = 0.0,
        dense_model: str = "",
    ) -> None:
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Index(docs)
        self.dense = DenseTextIndex(docs, model_name=dense_model) if dense_weight > 0 else None

    def scores(self, query: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bm25_scores = normalize_scores(self.bm25.scores(query))
        dense_scores = normalize_scores(self.dense.scores(query)) if self.dense is not None else np.zeros_like(bm25_scores)
        combined = self.bm25_weight * bm25_scores + self.dense_weight * dense_scores
        return combined, bm25_scores, dense_scores


def _concept_candidates(
    taxonomy: list[TaxonomyConcept],
    scores: dict[int, float],
    bm25_scores: dict[int, float],
    dense_scores: dict[int, float],
    memory_boosts: dict[int, float] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    boosts = memory_boosts or {}
    return [
        {
            "rank": rank,
            "tag": taxonomy[idx].tag,
            "entity_type": taxonomy[idx].entity_type,
            "text": taxonomy[idx].text,
            "score": float(score),
            "bm25_score": float(bm25_scores.get(idx, 0.0)),
            "dense_score": float(dense_scores.get(idx, 0.0)),
            "memory_boost": float(boosts.get(idx, 0.0)),
        }
        for rank, (idx, score) in enumerate(ranked, start=1)
    ]


def _descending_score_order(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.size == 0 or top_k <= 0:
        return np.zeros(0, dtype=int)
    return np.lexsort((np.arange(scores.size), -scores))[:top_k]


class LTMRetriever:
    """Long-term memory for taxonomy retrieval and past examples."""

    def __init__(
        self,
        taxonomy: list[TaxonomyConcept],
        memory_df: pd.DataFrame,
        bm25_weight: float = 1.0,
        dense_weight: float = 0.0,
        dense_model: str = "",
        taxonomy_doc_mode: str = "full",
    ) -> None:
        self.taxonomy = taxonomy
        self.memory_df = memory_df.copy()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.by_type = self._index_concepts_by_type(taxonomy)
        self.tag_to_taxonomy_idx = {concept.tag: idx for idx, concept in enumerate(taxonomy)}

        taxonomy_docs = [taxonomy_document(concept, taxonomy_doc_mode) for concept in taxonomy]
        self.taxonomy_index = HybridTextIndex(taxonomy_docs, bm25_weight, dense_weight, dense_model)

        self.memory_df["evidence"] = [
            rewrite_evidence_for_retrieval(
                localize_context(row.context, row.category, row.entity),
                row.category,
                row.entity,
                row.entity_type,
            )
            for row in self.memory_df.itertuples(index=False)
        ]
        memory_docs = [
            row.evidence
            for row in self.memory_df.itertuples(index=False)
        ]
        self.memory_index = HybridTextIndex(memory_docs, bm25_weight, dense_weight)

    @staticmethod
    def _index_concepts_by_type(taxonomy: list[TaxonomyConcept]) -> dict[str, list[int]]:
        by_type: dict[str, list[int]] = {}
        for idx, concept in enumerate(taxonomy):
            by_type.setdefault(concept.entity_type, []).append(idx)
        return by_type

    def retrieve(
        self,
        entity: Any,
        entity_type: str,
        evidence: str,
        top_k: int = 500,
        memory_k: int = 8,
        memory_weight: float = 0.10,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        query = evidence
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        candidate_scores, bm25_scores, dense_scores = self._taxonomy_scores(query, allowed)
        memory_hits = self._memory_hits(query, entity_type, memory_k)
        memory_boosts: dict[int, float] = {}

        for hit in memory_hits:
            tax_idx = self.tag_to_taxonomy_idx.get(hit["answer"])
            if tax_idx is not None:
                boost = memory_weight * hit["score"]
                candidate_scores[tax_idx] = candidate_scores.get(tax_idx, 0.0) + boost
                memory_boosts[tax_idx] = memory_boosts.get(tax_idx, 0.0) + boost

        candidates = _concept_candidates(
            self.taxonomy,
            candidate_scores,
            bm25_scores,
            dense_scores,
            memory_boosts,
            top_k,
        )
        return candidates, memory_hits

    def retrieve_taxonomy(self, entity_type: str, evidence: str, top_k: int = 500) -> list[dict[str, Any]]:
        query = evidence
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        candidate_scores, bm25_scores, dense_scores = self._taxonomy_scores(query, allowed)
        return _concept_candidates(self.taxonomy, candidate_scores, bm25_scores, dense_scores, {}, top_k)

    def _taxonomy_scores(
        self,
        query: str,
        allowed: list[int],
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        _hybrid, bm25, dense = self.taxonomy_index.scores(query)
        allowed_array = np.array(allowed, dtype=int)
        if allowed_array.size:
            bm25_allowed = normalize_scores(bm25[allowed_array])
            dense_allowed = normalize_scores(dense[allowed_array])
            hybrid_allowed = self.bm25_weight * bm25_allowed + self.dense_weight * dense_allowed
        else:
            bm25_allowed = np.zeros(0, dtype=float)
            dense_allowed = np.zeros(0, dtype=float)
            hybrid_allowed = np.zeros(0, dtype=float)
        scores = {idx: float(hybrid_allowed[pos]) for pos, idx in enumerate(allowed)}
        bm25_scores = {idx: float(bm25_allowed[pos]) for pos, idx in enumerate(allowed)}
        dense_scores = {idx: float(dense_allowed[pos]) for pos, idx in enumerate(allowed)}
        return scores, bm25_scores, dense_scores

    def _memory_hits(self, query: str, entity_type: str, memory_k: int) -> list[dict[str, Any]]:
        mem_sims, bm25_scores, dense_scores = self.memory_index.scores(query)
        mem_order = _descending_score_order(mem_sims, memory_k)
        hits: list[dict[str, Any]] = []
        for mem_idx in mem_order:
            sim = float(mem_sims[mem_idx])
            if sim <= 0:
                continue
            mem_row = self.memory_df.iloc[int(mem_idx)]
            if mem_row["entity_type"] != entity_type:
                continue
            hits.append(
                {
                    "row_index": int(mem_idx),
                    "score": sim,
                    "entity": str(mem_row["entity"]),
                    "entity_type": mem_row["entity_type"],
                    "answer": mem_row["answer"],
                    "bm25_score": float(bm25_scores[mem_idx]),
                    "dense_score": float(dense_scores[mem_idx]),
                }
            )
        return hits


class DynamicLTMRetriever:
    """Taxonomy retrieval plus event-driven LTM boosts.

    Unlike ``LTMRetriever``, this class does not freeze selector memory at
    startup. It reads the current LTM store at each call, so writes from the
    validator-corrector can influence future samples.
    """

    def __init__(
        self,
        taxonomy: list[TaxonomyConcept],
        ltm: LTMStore,
        bm25_weight: float = 1.0,
        dense_weight: float = 0.0,
        dense_model: str = "",
        taxonomy_doc_mode: str = "full",
    ) -> None:
        self.taxonomy = taxonomy
        self.ltm = ltm
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.by_type = LTMRetriever._index_concepts_by_type(taxonomy)
        self.tag_to_taxonomy_idx = {concept.tag: idx for idx, concept in enumerate(taxonomy)}
        taxonomy_docs = [taxonomy_document(concept, taxonomy_doc_mode) for concept in taxonomy]
        self.taxonomy_index = HybridTextIndex(taxonomy_docs, bm25_weight, dense_weight, dense_model)

    def retrieve(
        self,
        entity: Any,
        entity_type: str,
        evidence: str,
        top_k: int = 500,
        memory_k: int = 8,
        memory_weight: float = 0.10,
        error_weight: float = 0.05,
        table_pattern_weight: float = 0.05,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        query = evidence
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        candidate_scores, bm25_scores, dense_scores = self._taxonomy_scores(query, allowed)
        memory_boosts: dict[int, float] = {}

        memory_hits = self._rank_memory("selector_memory", query, entity_type, memory_k)
        error_hits = self._rank_memory("error_book", query, entity_type, memory_k)
        table_pattern_hits = self._rank_memory("table_context_patterns", query, entity_type, memory_k)

        self._apply_tag_boost(candidate_scores, memory_boosts, memory_hits, "tag", memory_weight)
        self._apply_tag_boost(candidate_scores, memory_boosts, error_hits, "correct_tag", error_weight)
        self._apply_tag_boost(candidate_scores, memory_boosts, table_pattern_hits, "tag", table_pattern_weight)

        candidates = _concept_candidates(
            self.taxonomy,
            candidate_scores,
            bm25_scores,
            dense_scores,
            memory_boosts,
            top_k,
        )
        return candidates, {
            "selector_memory": memory_hits,
            "error_book": error_hits,
            "table_context_patterns": table_pattern_hits,
        }

    def retrieve_taxonomy(self, entity_type: str, evidence: str, top_k: int = 500) -> list[dict[str, Any]]:
        query = evidence
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        candidate_scores, bm25_scores, dense_scores = self._taxonomy_scores(query, allowed)
        return _concept_candidates(self.taxonomy, candidate_scores, bm25_scores, dense_scores, {}, top_k)

    def retrieve_table_patterns_for_evidence(
        self,
        context: str,
        category: str,
        entity: Any,
        entity_type: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if category != "table" or top_k <= 0:
            return []
        query_evidence = localize_context(context, "table", entity, max_chars=2500)
        query_evidence = rewrite_evidence_for_retrieval(query_evidence, category, entity, entity_type)
        query = query_evidence
        return self._rank_memory("table_context_patterns", query, entity_type, top_k)

    def _taxonomy_scores(
        self,
        query: str,
        allowed: list[int],
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        _hybrid, bm25, dense = self.taxonomy_index.scores(query)
        allowed_array = np.array(allowed, dtype=int)
        if allowed_array.size:
            bm25_allowed = normalize_scores(bm25[allowed_array])
            dense_allowed = normalize_scores(dense[allowed_array])
            hybrid_allowed = self.bm25_weight * bm25_allowed + self.dense_weight * dense_allowed
        else:
            bm25_allowed = np.zeros(0, dtype=float)
            dense_allowed = np.zeros(0, dtype=float)
            hybrid_allowed = np.zeros(0, dtype=float)
        scores = {idx: float(hybrid_allowed[pos]) for pos, idx in enumerate(allowed)}
        bm25_scores = {idx: float(bm25_allowed[pos]) for pos, idx in enumerate(allowed)}
        dense_scores = {idx: float(dense_allowed[pos]) for pos, idx in enumerate(allowed)}
        return scores, bm25_scores, dense_scores

    def _rank_memory(
        self,
        namespace: str,
        query: str,
        entity_type: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        records = [record for record in self.ltm.records(namespace) if record.get("entity_type") == entity_type]
        if not records:
            return []
        docs = [self._memory_doc(record) for record in records]
        index = HybridTextIndex(docs, self.bm25_weight, self.dense_weight)
        sims, bm25_scores, dense_scores = index.scores(query)
        order = _descending_score_order(sims, top_k)
        hits: list[dict[str, Any]] = []
        for pos in order:
            score = float(sims[pos])
            if score <= 0:
                continue
            hit = dict(records[int(pos)])
            hit["score"] = score
            hit["bm25_score"] = float(bm25_scores[pos])
            hit["dense_score"] = float(dense_scores[pos])
            hits.append(hit)
        return hits

    @staticmethod
    def _memory_doc(record: dict[str, Any]) -> str:
        parts = [
            record.get("entity", ""),
            record.get("entity_type", ""),
            record.get("evidence", ""),
            record.get("pattern", ""),
            record.get("reason", ""),
            record.get("lesson", ""),
            record.get("tag", ""),
            record.get("correct_tag", ""),
        ]
        return normalize_space(" ".join(str(part) for part in parts))

    def _apply_tag_boost(
        self,
        candidate_scores: dict[int, float],
        memory_boosts: dict[int, float],
        hits: list[dict[str, Any]],
        tag_field: str,
        weight: float,
    ) -> None:
        for hit in hits:
            tag = hit.get(tag_field)
            tax_idx = self.tag_to_taxonomy_idx.get(tag)
            if tax_idx is not None:
                boost = weight * float(hit["score"])
                candidate_scores[tax_idx] = candidate_scores.get(tax_idx, 0.0) + boost
                memory_boosts[tax_idx] = memory_boosts.get(tax_idx, 0.0) + boost
