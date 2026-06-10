# Computes tag accuracy and Agent-1 retrieval recall metrics.
from __future__ import annotations

import math
from typing import Any


Record = dict[str, Any]


def evaluate_records(
    records: list[Record],
    recall_k: tuple[int, ...],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate final prediction accuracy and Agent-1 BM25 recall."""

    metrics: dict[str, Any] = {"num_examples": len(records)}
    if metadata:
        metrics.update(metadata)

    scored = [record for record in records if record.get("gold", {}).get("Tag")]
    if not scored:
        metrics["tag_accuracy"] = math.nan
        metrics["recall_at_k"] = {str(k): math.nan for k in recall_k}
        return metrics

    correct = sum(1 for record in scored if record.get("correct"))
    recall_counts = {k: 0 for k in recall_k}
    for record in scored:
        gold_tag = record["gold"]["Tag"]
        candidate_tags = [candidate["tag"] for candidate in record["retrieval"]["top_k"]]
        for k in recall_k:
            recall_counts[k] += int(gold_tag in candidate_tags[:k])

    metrics["scored_examples"] = len(scored)
    metrics["tag_accuracy"] = correct / len(scored)
    metrics["recall_at_k"] = {str(k): recall_counts[k] / len(scored) for k in recall_k}
    return metrics
