# Implements deterministic BM25 retrieval for taxonomy concepts and LTM memories.
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from .data import TaxonomyConcept


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
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
        if not self.doc_tokens or self.avg_doc_length <= 0:
            return scores

        for token in sorted(set(tokenize(query))):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * freq * (self.k1 + 1.0) / denom
        return scores

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = np.lexsort((np.arange(scores.size), -scores))[:top_k]
        return [(int(idx), float(scores[idx])) for idx in order]


class TaxonomyBM25:
    """Agent-1 retrieval index: type filter, then BM25 over taxonomy text."""

    def __init__(self, taxonomy: list[TaxonomyConcept]) -> None:
        self.taxonomy = taxonomy
        self.by_type: dict[str, list[int]] = {}
        for idx, concept in enumerate(taxonomy):
            self.by_type.setdefault(concept.entity_type, []).append(idx)
        self.index_by_type = {
            entity_type: BM25Index([taxonomy[idx].text for idx in indices])
            for entity_type, indices in self.by_type.items()
        }

    def retrieve(self, context: str, entity_type: str, top_k: int) -> list[dict[str, Any]]:
        indices = self.by_type.get(entity_type, [])
        index = self.index_by_type.get(entity_type)
        if index is None:
            return []

        candidates: list[dict[str, Any]] = []
        for rank, (local_idx, score) in enumerate(index.rank(context, top_k), start=1):
            tax_idx = indices[local_idx]
            concept = self.taxonomy[tax_idx]
            candidates.append(
                {
                    "rank": rank,
                    "tag": concept.tag,
                    "entity_type": concept.entity_type,
                    "text": concept.text,
                    "score": score,
                }
            )
        return candidates


def rank_memory(records: list[dict[str, Any]], context: str, entity_type: str, top_k: int) -> list[dict[str, Any]]:
    typed_records = [record for record in records if record.get("entity_type") == entity_type]
    if not typed_records:
        return []

    index = BM25Index([str(record.get("context", "")) for record in typed_records])
    hits: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(index.rank(context, top_k), start=1):
        hit = dict(typed_records[idx])
        hit["rank"] = rank
        hit["score"] = score
        hits.append(hit)
    return hits
