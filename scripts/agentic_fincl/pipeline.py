from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_taxonomy
from .evaluation import evaluate_fixed_memory_records, write_fixed_memory_breakdown
from .rerankers import LlamaReranker, RetrievalTop1Reranker, Reranker
from .retrieval import LTMRetriever
from .text_utils import localize_context, rewrite_evidence_for_retrieval
from .validation import validate_prediction


@dataclass(frozen=True)
class PipelineConfig:
    memory_csv: Path
    test_csv: Path
    taxonomy_jsonl: Path
    output_dir: Path
    top_k: int = 200
    rerank_k: int = 200
    memory_k: int = 8
    bm25_weight: float = 1.0
    dense_weight: float = 0.0
    dense_model: str = ""
    taxonomy_doc_mode: str = "full"
    memory_weight: float = 0.10
    save_top_k: int = 200
    recall_k: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)
    reranker: str = "retrieval"
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    limit: int = 0


def build_reranker(config: PipelineConfig) -> Reranker:
    if config.reranker == "retrieval":
        return RetrievalTop1Reranker()
    if config.reranker == "llama":
        return LlamaReranker(config.model)
    raise ValueError(f"Unsupported reranker: {config.reranker}")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    memory_df = pd.read_csv(config.memory_csv)
    test_df = pd.read_csv(config.test_csv)
    taxonomy = load_taxonomy(config.taxonomy_jsonl)
    retriever = LTMRetriever(
        taxonomy,
        memory_df,
        bm25_weight=config.bm25_weight,
        dense_weight=config.dense_weight,
        dense_model=config.dense_model,
        taxonomy_doc_mode=config.taxonomy_doc_mode,
    )
    reranker = build_reranker(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "predictions.jsonl"

    records: list[dict[str, Any]] = []

    with predictions_path.open("w", encoding="utf-8") as f:
        for row_idx, row in enumerate(test_df.itertuples(index=False), start=1):
            record = run_one(row_idx, row, retriever, reranker, config)
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if config.limit and row_idx >= config.limit:
                break

    metrics = evaluate_fixed_memory_records(
        records,
        config.recall_k,
        metadata={
            "reranker": config.reranker,
            "model": config.model if config.reranker == "llama" else None,
            "top_k": config.top_k,
            "rerank_k": config.rerank_k if config.reranker == "llama" else None,
            "retrieval": "bm25" if config.dense_weight <= 0 else "hybrid_bm25_dense",
            "bm25_weight": config.bm25_weight,
            "dense_weight": config.dense_weight,
            "dense_model": (config.dense_model or "svd_fallback") if config.dense_weight > 0 else None,
            "taxonomy_doc_mode": config.taxonomy_doc_mode,
            "memory_k": config.memory_k,
            "memory_weight": config.memory_weight,
        },
    )
    metrics["predictions_path"] = str(predictions_path)
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_fixed_memory_breakdown(records, config.output_dir)
    return metrics


def run_one(
    row_idx: int,
    row: Any,
    retriever: LTMRetriever,
    reranker: Reranker,
    config: PipelineConfig,
) -> dict[str, Any]:
    # Step 1 and Step 2: target-centered entity/type record from the clean split.
    # The open-ended all-entity extractor can replace this block later.
    localized = localize_context(row.context, row.category, row.entity)
    evidence = rewrite_evidence_for_retrieval(localized, row.category, row.entity, row.entity_type)
    entity_record = {"value": str(row.entity), "type": row.entity_type}
    raw_candidates = retriever.retrieve_taxonomy(row.entity_type, evidence, top_k=config.top_k)

    # Step 3: LTM retrieval plus either top-1 selection or Llama reranking.
    candidates, memory_hits = retriever.retrieve(
        row.entity,
        row.entity_type,
        evidence,
        top_k=config.top_k,
        memory_k=config.memory_k,
        memory_weight=config.memory_weight,
    )
    selected_tag = reranker.choose(row.entity, row.entity_type, evidence, candidates[: config.rerank_k])

    # Step 4: validation flags for consistency and ambiguity.
    validation = validate_prediction(row.entity_type, candidates, selected_tag)
    is_correct = selected_tag == row.answer

    stm = {
        "context_id": f"test-{row_idx}",
        "category": row.category,
        "entity": entity_record,
        "evidence": evidence,
        "raw_top_k": raw_candidates[: config.save_top_k],
        "top_k": candidates[: config.save_top_k],
        "selected_concept": selected_tag,
        "validation": validation,
        "memory_hits": memory_hits[:5],
    }
    return {
        "row_index": row_idx - 1,
        "gold": {"Fact": str(row.entity), "Type": row.entity_type, "Tag": row.answer},
        "prediction": {"Fact": str(row.entity), "Type": row.entity_type, "Tag": selected_tag},
        "correct": is_correct,
        "stm": stm,
    }
