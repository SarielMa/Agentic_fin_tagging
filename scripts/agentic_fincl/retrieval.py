from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import TaxonomyConcept, tag_terms
from .memory_store import LTMStore
from .text_utils import build_query_text, localize_context, normalize_space


class LTMRetriever:
    """Long-term memory for taxonomy retrieval and past examples."""

    def __init__(self, taxonomy: list[TaxonomyConcept], memory_df: pd.DataFrame) -> None:
        self.taxonomy = taxonomy
        self.memory_df = memory_df.copy()
        self.by_type = self._index_concepts_by_type(taxonomy)
        self.tag_to_taxonomy_idx = {concept.tag: idx for idx, concept in enumerate(taxonomy)}

        taxonomy_docs = [
            normalize_space(f"{concept.text} {tag_terms(concept.tag)} {concept.entity_type}")
            for concept in taxonomy
        ]
        self.taxonomy_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        self.taxonomy_matrix = self.taxonomy_vectorizer.fit_transform(taxonomy_docs)

        self.memory_df["evidence"] = [
            localize_context(row.context, row.category, row.entity)
            for row in self.memory_df.itertuples(index=False)
        ]
        memory_docs = [
            build_query_text(row.entity, row.entity_type, row.evidence)
            for row in self.memory_df.itertuples(index=False)
        ]
        self.memory_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        self.memory_matrix = self.memory_vectorizer.fit_transform(memory_docs)

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
        top_k: int = 200,
        memory_k: int = 8,
        memory_weight: float = 0.10,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        query = build_query_text(entity, entity_type, evidence)
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        candidate_scores = self._taxonomy_scores(query, allowed)
        memory_hits = self._memory_hits(query, entity_type, memory_k)

        for hit in memory_hits:
            tax_idx = self.tag_to_taxonomy_idx.get(hit["answer"])
            if tax_idx is not None:
                candidate_scores[tax_idx] = candidate_scores.get(tax_idx, 0.0) + memory_weight * hit["score"]

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidates = [
            {
                "rank": rank,
                "tag": self.taxonomy[idx].tag,
                "entity_type": self.taxonomy[idx].entity_type,
                "text": self.taxonomy[idx].text,
                "score": float(score),
            }
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]
        return candidates, memory_hits

    def _taxonomy_scores(self, query: str, allowed: list[int]) -> dict[int, float]:
        q_tax = self.taxonomy_vectorizer.transform([query])
        sims = cosine_similarity(q_tax, self.taxonomy_matrix[allowed]).ravel()
        return {allowed[pos]: float(score) for pos, score in enumerate(sims)}

    def _memory_hits(self, query: str, entity_type: str, memory_k: int) -> list[dict[str, Any]]:
        q_mem = self.memory_vectorizer.transform([query])
        mem_sims = cosine_similarity(q_mem, self.memory_matrix).ravel()
        mem_order = np.argsort(-mem_sims)[:memory_k]
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
                }
            )
        return hits


class DynamicLTMRetriever:
    """Taxonomy retrieval plus event-driven LTM boosts.

    Unlike ``LTMRetriever``, this class does not freeze selector memory at
    startup. It reads the current LTM store at each call, so writes from the
    validator-corrector can influence future samples.
    """

    def __init__(self, taxonomy: list[TaxonomyConcept], ltm: LTMStore) -> None:
        self.taxonomy = taxonomy
        self.ltm = ltm
        self.by_type = LTMRetriever._index_concepts_by_type(taxonomy)
        self.tag_to_taxonomy_idx = {concept.tag: idx for idx, concept in enumerate(taxonomy)}
        taxonomy_docs = [
            normalize_space(f"{concept.text} {tag_terms(concept.tag)} {concept.entity_type}")
            for concept in taxonomy
        ]
        self.taxonomy_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        self.taxonomy_matrix = self.taxonomy_vectorizer.fit_transform(taxonomy_docs)

    def retrieve(
        self,
        entity: Any,
        entity_type: str,
        evidence: str,
        top_k: int = 200,
        memory_k: int = 8,
        memory_weight: float = 0.10,
        error_weight: float = 0.05,
        table_pattern_weight: float = 0.05,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        query = build_query_text(entity, entity_type, evidence)
        allowed = self.by_type.get(entity_type, list(range(len(self.taxonomy))))
        candidate_scores = self._taxonomy_scores(query, allowed)

        memory_hits = self._rank_memory("selector_memory", query, entity_type, memory_k)
        error_hits = self._rank_memory("error_book", query, entity_type, memory_k)
        table_pattern_hits = self._rank_memory("table_context_patterns", query, entity_type, memory_k)

        self._apply_tag_boost(candidate_scores, memory_hits, "tag", memory_weight)
        self._apply_tag_boost(candidate_scores, error_hits, "correct_tag", error_weight)
        self._apply_tag_boost(candidate_scores, table_pattern_hits, "tag", table_pattern_weight)

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidates = [
            {
                "rank": rank,
                "tag": self.taxonomy[idx].tag,
                "entity_type": self.taxonomy[idx].entity_type,
                "text": self.taxonomy[idx].text,
                "score": float(score),
            }
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]
        return candidates, {
            "selector_memory": memory_hits,
            "error_book": error_hits,
            "table_context_patterns": table_pattern_hits,
        }

    def _taxonomy_scores(self, query: str, allowed: list[int]) -> dict[int, float]:
        q_tax = self.taxonomy_vectorizer.transform([query])
        sims = cosine_similarity(q_tax, self.taxonomy_matrix[allowed]).ravel()
        return {allowed[pos]: float(score) for pos, score in enumerate(sims)}

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
        vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(docs)
        q_mem = vectorizer.transform([query])
        sims = cosine_similarity(q_mem, matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        hits: list[dict[str, Any]] = []
        for pos in order:
            score = float(sims[pos])
            if score <= 0:
                continue
            hit = dict(records[int(pos)])
            hit["score"] = score
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
        hits: list[dict[str, Any]],
        tag_field: str,
        weight: float,
    ) -> None:
        for hit in hits:
            tag = hit.get(tag_field)
            tax_idx = self.tag_to_taxonomy_idx.get(tag)
            if tax_idx is not None:
                candidate_scores[tax_idx] = candidate_scores.get(tax_idx, 0.0) + weight * float(hit["score"])
