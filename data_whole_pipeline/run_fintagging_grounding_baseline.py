#!/usr/bin/env python3
"""Grounding experiments for FinTagging context-aware tag selection.

The direct retrieval method has two stages:

1. BM25 retrieval over enriched US-GAAP concept definitions.
2. Optional Qwen reranking over the retrieved candidates.

The one-pass grounding method first asks an LLM to generate a brief concept
description from the source context, entity, and type, then uses that description
with the entity/type as the BM25 query. Candidate reranking and evaluation are
shared with direct retrieval.

Retrieval-only stages can run on CPU. LLM query generation and reranking require
the model backend selected by the command line.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ImportError:  # pragma: no cover - torch is only required for model generation.
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_JSONL = SCRIPT_DIR / "FinTagging_800_200_grounding_test_JSON" / "data" / "test.jsonl"
DEFAULT_TAXONOMY_JSONL = (
    PROJECT_ROOT / "retrieval_data" / "us_gaap_2024_enriched" / "us_gaap_2024_enriched_retrieval.jsonl"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_fintagging_grounding_baseline" / "qwen3_32b_direct_retrieval"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
TAG_PREFIX = "us-gaap:"
QUERY_MODE_ALIASES = {
    "direct": "direct_retrieval",
    "direct_retrieval": "direct_retrieval",
    "llm_description": "one_pass_grounding",
    "one_pass_grounding": "one_pass_grounding",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = normalize_space(data)
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return normalize_space(" ".join(self.parts))


@dataclass(frozen=True)
class Concept:
    tag: str
    raw_tag: str
    entity_type: str
    standard_label: str
    documentation: str
    references: list[str]
    retrieval_text: str


@dataclass(frozen=True)
class Example:
    example_idx: int
    context_id: Any
    source_sample_idx: Any
    input_type: str
    entity: str
    entity_type: str
    original_context: str
    query_context: str
    gold_tags: list[str]


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def normalize_tag(tag: Any) -> str:
    text = normalize_space(tag)
    if not text:
        return ""
    return text if text.startswith(TAG_PREFIX) else f"{TAG_PREFIX}{text}"


def raw_tag(tag: Any) -> str:
    text = normalize_space(tag)
    return text[len(TAG_PREFIX) :] if text.startswith(TAG_PREFIX) else text


def canonical_query_mode(value: Any) -> str:
    text = normalize_space(value)
    canonical = QUERY_MODE_ALIASES.get(text)
    if canonical is None:
        raise ValueError(
            f"Unsupported query_mode={text}. Expected one of: {sorted(QUERY_MODE_ALIASES)}"
        )
    return canonical


def tokenize(text: Any) -> list[str]:
    expanded = CAMEL_BOUNDARY_RE.sub(" ", str(text))
    tokens: list[str] = []
    for token in TOKEN_RE.findall(f"{text} {expanded}"):
        normalized = normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)
    return tokens


def normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def html_to_visible_text(text: str) -> str:
    if "<" not in text or ">" not in text:
        return normalize_space(unescape(text))
    parser = VisibleTextParser()
    try:
        parser.feed(text)
        parsed = parser.text()
    except Exception:
        parsed = ""
    return parsed or normalize_space(re.sub(r"<[^>]+>", " ", unescape(text)))


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head].rstrip()} ... {text[-tail:].lstrip()}"


def extract_balanced_json(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = cleaned.strip("`")
    for candidate in (cleaned, extract_balanced_json(cleaned, "{", "}")):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, True
    return {}, False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_examples(path: Path, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    for example_idx, row in enumerate(rows):
        fields = row.get("input_fields")
        if not isinstance(fields, dict):
            fields = json.loads(row.get("input", "{}"))
        original_context = normalize_space(fields.get("original_context", ""))
        visible_context = html_to_visible_text(original_context)
        entity = normalize_space(
            fields.get("numeric_entity")
            or fields.get("entity")
            or fields.get("value")
            or fields.get("numeric_value")
            or ""
        )
        entity_type = normalize_space(fields.get("datatype") or fields.get("type") or "")
        gold_tags = [normalize_tag(tag) for tag in row.get("ground_truth_concepts", []) if normalize_tag(tag)]
        examples.append(
            Example(
                example_idx=example_idx,
                context_id=row.get("context_id"),
                source_sample_idx=row.get("source_sample_idx"),
                input_type=normalize_space(row.get("input_type", "")),
                entity=entity,
                entity_type=entity_type,
                original_context=original_context,
                query_context=visible_context,
                gold_tags=gold_tags,
            )
        )
    return examples


def load_taxonomy(path: Path) -> list[Concept]:
    concepts: list[Concept] = []
    for row in load_jsonl(path):
        tag = normalize_tag(row["tag"])
        concepts.append(
            Concept(
                tag=tag,
                raw_tag=raw_tag(tag),
                entity_type=normalize_space(row.get("type", "")),
                standard_label=normalize_space(row.get("standard_label", "")),
                documentation=normalize_space(row.get("documentation", "")),
                references=[normalize_space(ref) for ref in row.get("references", []) if normalize_space(ref)],
                retrieval_text=normalize_space(row.get("retrieval_text", "")),
            )
        )
    return concepts


class BM25Index:
    def __init__(
        self,
        concepts: list[Concept],
        k1: float = 1.5,
        b: float = 0.75,
        include_type_in_text: bool = False,
    ) -> None:
        self.concepts = concepts
        self.k1 = k1
        self.b = b
        self.doc_tokens = [
            tokenize(
                f"{concept.entity_type}. {concept.retrieval_text}"
                if include_type_in_text
                else concept.retrieval_text
            )
            for concept in concepts
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()

        for doc_idx, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            doc_freq.update(counts.keys())
            for token, freq in counts.items():
                self.postings[token].append((doc_idx, freq))

        n_docs = len(self.doc_tokens)
        self.idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.doc_tokens or self.avg_doc_length <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(tokenize(query))):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * freq * (self.k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < min(top_k, len(self.concepts)):
            seen = {idx for idx, _ in ranked}
            ranked.extend((idx, 0.0) for idx in range(len(self.concepts)) if idx not in seen)
        return ranked[:top_k]


class TaxonomyRetriever:
    def __init__(self, concepts: list[Concept], type_filter: bool) -> None:
        self.type_filter = type_filter
        self.all_index = BM25Index(concepts, include_type_in_text=True)
        self.by_type: dict[str, list[Concept]] = defaultdict(list)
        for concept in concepts:
            self.by_type[concept.entity_type].append(concept)
        self.index_by_type = {
            entity_type: BM25Index(type_concepts, include_type_in_text=False)
            for entity_type, type_concepts in self.by_type.items()
        }

    def retrieve(self, query: str, entity_type: str, top_k: int) -> list[tuple[Concept, float]]:
        if self.type_filter and entity_type in self.index_by_type:
            index = self.index_by_type[entity_type]
        else:
            index = self.all_index
        return [(index.concepts[idx], score) for idx, score in index.rank(query, top_k)]


def build_direct_query(example: Example) -> str:
    return normalize_space(f"{example.entity} {example.entity_type} {example.query_context}")


def build_retrieval_query(
    example: Example,
    query_mode: str,
    query_descriptions: dict[int, str] | None = None,
) -> tuple[str, str | None]:
    query_mode = canonical_query_mode(query_mode)
    if query_mode == "direct_retrieval":
        return build_direct_query(example), None

    if query_mode != "one_pass_grounding":
        raise ValueError(f"Unsupported query_mode={query_mode}")

    description = normalize_space((query_descriptions or {}).get(example.example_idx, ""))
    if not description:
        description = build_direct_query(example)
    return normalize_space(f"{example.entity} {example.entity_type} {description}"), description


def concept_to_candidate(concept: Concept, rank: int, score: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "tag": concept.tag,
        "type": concept.entity_type,
        "standard_label": concept.standard_label,
        "documentation": concept.documentation,
        "references": concept.references,
        "retrieval_text": concept.retrieval_text,
        "bm25_score": round(score, 8),
    }


def first_gold_rank(ranking: list[str], gold_tags: Iterable[str]) -> int | None:
    gold = set(gold_tags)
    for idx, tag in enumerate(ranking, start=1):
        if normalize_tag(tag) in gold:
            return idx
    return None


def metric_row(ranking: list[str], gold_tags: list[str], top_ks: tuple[int, ...]) -> dict[str, Any]:
    rank = first_gold_rank(ranking, gold_tags)
    row: dict[str, Any] = {
        "rank": rank,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "accuracy": bool(rank == 1),
    }
    for k in top_ks:
        row[f"recall_at_{k}"] = bool(rank is not None and rank <= k)
    return row


def aggregate_metric_rows(rows: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    metrics: dict[str, Any] = {
        "n": n,
        "accuracy": round(sum(bool(row["accuracy"]) for row in rows) / n, 6),
        "mrr": round(sum(float(row["mrr"]) for row in rows) / n, 6),
    }
    for k in top_ks:
        metrics[f"recall_at_{k}"] = round(
            sum(bool(row[f"recall_at_{k}"]) for row in rows) / n,
            6,
        )
    return metrics


def build_candidate_records(
    examples: list[Example],
    taxonomy: list[Concept],
    top_k: int,
    type_filter: bool,
    query_mode: str,
    query_descriptions: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    query_mode = canonical_query_mode(query_mode)
    retriever = TaxonomyRetriever(taxonomy, type_filter=type_filter)
    records: list[dict[str, Any]] = []
    for example in examples:
        query, query_description = build_retrieval_query(example, query_mode, query_descriptions)
        candidates = [
            concept_to_candidate(concept, rank, score)
            for rank, (concept, score) in enumerate(
                retriever.retrieve(query, example.entity_type, top_k),
                start=1,
            )
        ]
        candidate_tags = [candidate["tag"] for candidate in candidates]
        retrieval_metrics = metric_row(candidate_tags, example.gold_tags, (10, 50, top_k))
        search_coverage = any(tag in set(candidate_tags) for tag in example.gold_tags)
        records.append(
            {
                "example_idx": example.example_idx,
                "context_id": example.context_id,
                "source_sample_idx": example.source_sample_idx,
                "input_type": example.input_type,
                "entity": example.entity,
                "type": example.entity_type,
                "gold_tags": example.gold_tags,
                "query_mode": query_mode,
                "query": query,
                "query_context": example.query_context,
                "query_description": query_description,
                "candidates": candidates,
                "candidate_union_tags": candidate_tags,
                "search_coverage": search_coverage,
                "retrieval_metrics": retrieval_metrics,
            }
        )
    return records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_predictions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    predictions: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        predictions[int(row["example_idx"])] = row
    return predictions


def candidate_query_mode(record: dict[str, Any]) -> str:
    return canonical_query_mode(record.get("query_mode") or "direct_retrieval")


def validate_candidate_records(records: list[dict[str, Any]], query_mode: str, top_k: int) -> None:
    query_mode = canonical_query_mode(query_mode)
    mismatches = [
        record.get("example_idx")
        for record in records
        if candidate_query_mode(record) != query_mode
    ]
    if mismatches:
        raise ValueError(
            f"Existing candidate file has query_mode={candidate_query_mode(records[0])}, "
            f"but requested query_mode={query_mode}. Set REUSE_CANDIDATES=0 or use a separate OUTPUT_DIR."
        )

    wrong_top_k = [
        record.get("example_idx")
        for record in records
        if len(record.get("candidates", [])) > top_k
    ]
    if wrong_top_k:
        raise ValueError(
            f"Existing candidate file contains more than top_k={top_k} candidates in at least one row. "
            "Set REUSE_CANDIDATES=0 to rebuild it."
        )


def format_candidate_for_prompt(candidate: dict[str, Any], doc_max_chars: int) -> str:
    docs = truncate_text(candidate.get("documentation", ""), doc_max_chars)
    refs = ", ".join(candidate.get("references", [])[:3])
    parts = [
        f"[{candidate['rank']}] {candidate['tag']}",
        f"Type: {candidate.get('type', '')}",
        f"Label: {candidate.get('standard_label', '')}",
    ]
    if docs:
        parts.append(f"Definition: {docs}")
    if refs:
        parts.append(f"References: {refs}")
    return "\n".join(parts)


def build_rerank_messages(
    record: dict[str, Any],
    context_max_chars: int,
    doc_max_chars: int,
    rerank_list_size: int,
) -> list[dict[str, str]]:
    context = truncate_text(str(record.get("query_context") or record.get("query", "")), context_max_chars)
    candidate_text = "\n\n".join(
        format_candidate_for_prompt(candidate, doc_max_chars)
        for candidate in record.get("candidates", [])
    )
    user = f"""Select the US-GAAP concept that best matches the entity in the source context.

Input:
Entity: {record.get("entity", "")}
Type: {record.get("type", "")}
Context:
{context}

Candidates:
{candidate_text}

Return JSON only with this schema:
{{"selected_index": 1, "selected_tag": "us-gaap:ExampleTag", "ranked_indices": [1, 2, 3]}}

Rules:
- Choose only from the candidate list.
- `selected_index` must be the index shown in square brackets.
- `ranked_indices` should contain up to {rerank_list_size} best candidate indices, best first.
- Do not include explanations or markdown."""
    return [
        {
            "role": "system",
            "content": "You are a precise US-GAAP XBRL concept grounding reranker.",
        },
        {"role": "user", "content": user},
    ]


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def build_prompt_under_token_budget(
    tokenizer: Any,
    record: dict[str, Any],
    context_max_chars: int,
    doc_max_chars: int,
    rerank_list_size: int,
    max_input_tokens: int,
) -> tuple[str, int, int, int]:
    context_options = [context_max_chars, min(context_max_chars, 8000), min(context_max_chars, 5000)]
    doc_options = [doc_max_chars, min(doc_max_chars, 240), min(doc_max_chars, 120), 0]
    tried: set[tuple[int, int]] = set()

    last_prompt = ""
    last_tokens = 0
    for ctx_chars in context_options:
        for doc_chars in doc_options:
            if (ctx_chars, doc_chars) in tried:
                continue
            tried.add((ctx_chars, doc_chars))
            messages = build_rerank_messages(record, ctx_chars, doc_chars, rerank_list_size)
            prompt = messages_to_prompt(tokenizer, messages)
            token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            last_prompt = prompt
            last_tokens = token_count
            if token_count <= max_input_tokens:
                return prompt, token_count, ctx_chars, doc_chars

    return last_prompt, last_tokens, min(context_options), min(doc_options)


def build_query_description_messages(example: Example, context_max_chars: int) -> list[dict[str, str]]:
    context = truncate_text(example.query_context, context_max_chars)
    user = f"""Write a brief retrieval query for finding the correct US-GAAP XBRL concept.

Input:
Entity value: {example.entity}
Type: {example.entity_type}
Source context:
{context}

Return JSON only with this schema:
{{"query": "brief concept description"}}

Rules:
- Describe the accounting concept represented by the entity in this context.
- Use wording likely to appear in US-GAAP labels or definitions.
- Do not name a specific US-GAAP tag unless it is explicitly present in the source context.
- Do not include explanations or markdown."""
    return [
        {
            "role": "system",
            "content": "You generate concise US-GAAP retrieval queries for XBRL concept grounding.",
        },
        {"role": "user", "content": user},
    ]


def build_query_description_prompt(
    tokenizer: Any,
    example: Example,
    context_max_chars: int,
    max_input_tokens: int,
) -> tuple[str, int, int]:
    context_options = [context_max_chars, min(context_max_chars, 8000), min(context_max_chars, 5000)]
    last_prompt = ""
    last_tokens = 0
    for ctx_chars in context_options:
        messages = build_query_description_messages(example, ctx_chars)
        prompt = messages_to_prompt(tokenizer, messages)
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        last_prompt = prompt
        last_tokens = token_count
        if token_count <= max_input_tokens:
            return prompt, token_count, ctx_chars
    return last_prompt, last_tokens, min(context_options)


def clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return normalize_space(text.strip('"').strip("'"))


def parse_query_description(raw_output: str, fallback: str) -> tuple[str, bool]:
    parsed, parse_ok = parse_json_object(raw_output)
    if parse_ok:
        for key in ("query", "description", "retrieval_query"):
            value = normalize_space(parsed.get(key))
            if value:
                return value, True

    cleaned = clean_model_text(raw_output)
    if cleaned:
        first_line = normalize_space(cleaned.splitlines()[0])
        if first_line:
            return first_line, False
    return fallback, False


def load_existing_query_descriptions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        records[int(row["example_idx"])] = row
    return records


def query_description_record(
    example: Example,
    raw_output: str,
    prompt_tokens: int,
    used_context_chars: int,
    backend: str,
    model_name: str,
) -> dict[str, Any]:
    fallback = build_direct_query(example)
    query_description, parse_ok = parse_query_description(raw_output, fallback)
    return {
        "example_idx": example.example_idx,
        "context_id": example.context_id,
        "source_sample_idx": example.source_sample_idx,
        "input_type": example.input_type,
        "entity": example.entity,
        "type": example.entity_type,
        "query_description": query_description,
        "parse_ok": parse_ok,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "used_context_max_chars": used_context_chars,
        "backend": backend,
        "model": model_name,
    }


def load_rerank_model(
    model_name: str,
    bf16: bool,
    trust_remote_code: bool,
    attn_implementation: str | None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_vllm_engine(args: argparse.Namespace, model_name: str) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": args.trust_remote_code,
        "dtype": "bfloat16" if args.bf16 else "float16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_input_tokens + max(args.max_new_tokens, args.query_max_new_tokens),
        "max_num_seqs": args.max_num_seqs,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True

    return tokenizer, LLM(**llm_kwargs)


def release_model_handles(*handles: Any) -> None:
    for handle in handles:
        del handle
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_query_descriptions(
    args: argparse.Namespace,
    examples: list[Example],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    if args.query_generation_backend == "vllm":
        return generate_query_descriptions_vllm(args, examples, output_path, tokenizer, llm)
    return generate_query_descriptions_transformers(args, examples, output_path)


def generate_query_descriptions_transformers(
    args: argparse.Namespace,
    examples: list[Example],
    output_path: Path,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_query_descriptions(output_path) if args.resume else {}
    records = dict(existing)
    pending = [example for example in examples if example.example_idx not in records]
    if not pending:
        return [records[example.example_idx] for example in examples if example.example_idx in records]

    tokenizer, model = load_rerank_model(
        args.query_generation_model,
        bf16=args.bf16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for offset, example in enumerate(pending, start=1):
            prompt, prompt_tokens, used_context_chars = build_query_description_prompt(
                tokenizer,
                example,
                context_max_chars=args.query_context_max_chars,
                max_input_tokens=args.query_max_input_tokens,
            )
            raw_output = generate_text(
                tokenizer,
                model,
                prompt,
                max_input_tokens=args.query_max_input_tokens,
                max_new_tokens=args.query_max_new_tokens,
                temperature=args.query_temperature,
                top_p=args.query_top_p,
            )
            record = query_description_record(
                example,
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                used_context_chars=used_context_chars,
                backend="transformers",
                model_name=args.query_generation_model,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records[example.example_idx] = record

            if offset % args.log_every == 0:
                print(f"Generated {offset}/{len(pending)} query descriptions")

    release_model_handles(model, tokenizer)

    return [records[example.example_idx] for example in examples if example.example_idx in records]


def generate_query_descriptions_vllm(
    args: argparse.Namespace,
    examples: list[Example],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_query_descriptions(output_path) if args.resume else {}
    records = dict(existing)
    pending = [example for example in examples if example.example_idx not in records]
    if not pending:
        return [records[example.example_idx] for example in examples if example.example_idx in records]

    loaded_here = tokenizer is None or llm is None
    if tokenizer is None or llm is None:
        tokenizer, llm = load_vllm_engine(args, args.query_generation_model)

    sampling_params = SamplingParams(
        temperature=args.query_temperature,
        top_p=args.query_top_p,
        max_tokens=args.query_max_new_tokens,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.vllm_batch_size):
            batch_examples = pending[start : start + args.vllm_batch_size]
            prompts: list[str] = []
            prompt_meta: list[tuple[int, int]] = []
            for example in batch_examples:
                prompt, prompt_tokens, used_context_chars = build_query_description_prompt(
                    tokenizer,
                    example,
                    context_max_chars=args.query_context_max_chars,
                    max_input_tokens=args.query_max_input_tokens,
                )
                prompts.append(prompt)
                prompt_meta.append((prompt_tokens, used_context_chars))

            outputs = llm.generate(prompts, sampling_params)
            for example, output, (prompt_tokens, used_context_chars) in zip(
                batch_examples,
                outputs,
                prompt_meta,
                strict=True,
            ):
                raw_output = output.outputs[0].text.strip() if output.outputs else ""
                record = query_description_record(
                    example,
                    raw_output=raw_output,
                    prompt_tokens=prompt_tokens,
                    used_context_chars=used_context_chars,
                    backend="vllm",
                    model_name=args.query_generation_model,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records[example.example_idx] = record

            handle.flush()
            processed = min(start + len(batch_examples), len(pending))
            if processed % args.log_every == 0 or processed == len(pending):
                print(f"Generated {processed}/{len(pending)} pending query descriptions with vLLM")

    if loaded_here:
        release_model_handles(llm, tokenizer)

    return [records[example.example_idx] for example in examples if example.example_idx in records]


def prediction_from_raw_output(
    record: dict[str, Any],
    raw_output: str,
    prompt_tokens: int,
    used_context_chars: int,
    used_doc_chars: int,
    backend: str,
) -> dict[str, Any]:
    parsed, parse_ok = parse_json_object(raw_output)
    model_ranked, selected_tag = parse_ranked_candidate_tags(parsed, record["candidates"])
    final_ranking = composite_ranking(model_ranked, record["candidates"])

    return {
        "example_idx": int(record["example_idx"]),
        "context_id": record.get("context_id"),
        "source_sample_idx": record.get("source_sample_idx"),
        "input_type": record.get("input_type"),
        "entity": record.get("entity"),
        "type": record.get("type"),
        "gold_tags": record.get("gold_tags", []),
        "selected_tag": final_ranking[0] if final_ranking else selected_tag,
        "model_ranked_tags": model_ranked,
        "final_ranking": final_ranking,
        "parse_ok": parse_ok,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "used_context_max_chars": used_context_chars,
        "used_candidate_doc_max_chars": used_doc_chars,
        "backend": backend,
        "retrieval_metrics": record.get("retrieval_metrics", {}),
        "search_coverage": record.get("search_coverage", False),
    }


def parse_ranked_candidate_tags(
    parsed: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[str], str | None]:
    by_index = {int(candidate["rank"]): candidate["tag"] for candidate in candidates}
    candidate_tags = {candidate["tag"] for candidate in candidates}
    ranked: list[str] = []

    for key in ("ranked_indices", "ranking", "ranked_candidate_indices"):
        value = parsed.get(key)
        if isinstance(value, list):
            for item in value:
                try:
                    tag = by_index.get(int(item))
                except (TypeError, ValueError):
                    tag = None
                if tag and tag not in ranked:
                    ranked.append(tag)

    for key in ("ranked_tags", "tags"):
        value = parsed.get(key)
        if isinstance(value, list):
            for item in value:
                tag = normalize_tag(item)
                if tag in candidate_tags and tag not in ranked:
                    ranked.append(tag)

    selected_tag: str | None = None
    try:
        selected_tag = by_index.get(int(parsed.get("selected_index")))
    except (TypeError, ValueError):
        selected_tag = None

    if selected_tag is None and parsed.get("selected_tag") is not None:
        candidate = normalize_tag(parsed.get("selected_tag"))
        if candidate in candidate_tags:
            selected_tag = candidate

    if selected_tag and selected_tag not in ranked:
        ranked.insert(0, selected_tag)
    elif selected_tag:
        ranked = [selected_tag] + [tag for tag in ranked if tag != selected_tag]

    return ranked, selected_tag


def composite_ranking(model_ranked: list[str], candidates: list[dict[str, Any]]) -> list[str]:
    candidate_order = [candidate["tag"] for candidate in candidates]
    candidate_set = set(candidate_order)
    ranking = []
    for tag in model_ranked:
        tag = normalize_tag(tag)
        if tag in candidate_set and tag not in ranking:
            ranking.append(tag)
    ranking.extend(tag for tag in candidate_order if tag not in ranking)
    return ranking


def rerank_records(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    if args.rerank_backend == "vllm":
        return rerank_records_vllm(args, records, output_path, tokenizer, llm)
    return rerank_records_transformers(args, records, output_path)


def rerank_records_transformers(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    tokenizer, model = load_rerank_model(
        args.rerank_model,
        bf16=args.bf16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_predictions(output_path) if args.resume else {}
    predictions = dict(existing)

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for offset, record in enumerate(records, start=1):
            example_idx = int(record["example_idx"])
            if example_idx in predictions:
                continue

            prompt, prompt_tokens, used_context_chars, used_doc_chars = build_prompt_under_token_budget(
                tokenizer,
                record,
                context_max_chars=args.context_max_chars,
                doc_max_chars=args.candidate_doc_max_chars,
                rerank_list_size=args.rerank_list_size,
                max_input_tokens=args.max_input_tokens,
            )
            raw_output = generate_text(
                tokenizer,
                model,
                prompt,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            prediction = prediction_from_raw_output(
                record,
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                used_context_chars=used_context_chars,
                used_doc_chars=used_doc_chars,
                backend="transformers",
            )
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            predictions[example_idx] = prediction

            if offset % args.log_every == 0:
                print(f"Reranked {offset}/{len(records)} records")

    return [predictions[int(record["example_idx"])] for record in records if int(record["example_idx"]) in predictions]


def rerank_records_vllm(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
    tokenizer: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    loaded_here = tokenizer is None or llm is None
    if tokenizer is None or llm is None:
        tokenizer, llm = load_vllm_engine(args, args.rerank_model)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_predictions(output_path) if args.resume else {}
    predictions = dict(existing)

    pending = [record for record in records if int(record["example_idx"]) not in predictions]
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.vllm_batch_size):
            batch_records = pending[start : start + args.vllm_batch_size]
            prompts: list[str] = []
            prompt_meta: list[tuple[int, int, int]] = []
            for record in batch_records:
                prompt, prompt_tokens, used_context_chars, used_doc_chars = build_prompt_under_token_budget(
                    tokenizer,
                    record,
                    context_max_chars=args.context_max_chars,
                    doc_max_chars=args.candidate_doc_max_chars,
                    rerank_list_size=args.rerank_list_size,
                    max_input_tokens=args.max_input_tokens,
                )
                prompts.append(prompt)
                prompt_meta.append((prompt_tokens, used_context_chars, used_doc_chars))

            outputs = llm.generate(prompts, sampling_params)
            for record, output, (prompt_tokens, used_context_chars, used_doc_chars) in zip(
                batch_records,
                outputs,
                prompt_meta,
                strict=True,
            ):
                raw_output = output.outputs[0].text.strip() if output.outputs else ""
                prediction = prediction_from_raw_output(
                    record,
                    raw_output=raw_output,
                    prompt_tokens=prompt_tokens,
                    used_context_chars=used_context_chars,
                    used_doc_chars=used_doc_chars,
                    backend="vllm",
                )
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                predictions[int(record["example_idx"])] = prediction

            handle.flush()
            processed = min(start + len(batch_records), len(pending))
            if processed % args.log_every == 0 or processed == len(pending):
                print(f"Reranked {processed}/{len(pending)} pending records with vLLM")

    if loaded_here:
        release_model_handles(llm, tokenizer)

    return [predictions[int(record["example_idx"])] for record in records if int(record["example_idx"]) in predictions]


def evaluate_records(
    candidate_records: list[dict[str, Any]],
    rerank_predictions: list[dict[str, Any]] | None,
    top_k: int,
    taxonomy_tags: set[str],
) -> dict[str, Any]:
    metrics = evaluate_record_scope(candidate_records, rerank_predictions, top_k, taxonomy_tags)
    metrics["by_input_type"] = {}
    input_types = sorted({normalize_space(record.get("input_type")) for record in candidate_records})
    for input_type in input_types:
        if not input_type:
            continue
        subset = [
            record
            for record in candidate_records
            if normalize_space(record.get("input_type")) == input_type
        ]
        metrics["by_input_type"][input_type] = evaluate_record_scope(
            subset,
            rerank_predictions,
            top_k,
            taxonomy_tags,
        )
    return metrics


def evaluate_record_scope(
    candidate_records: list[dict[str, Any]],
    rerank_predictions: list[dict[str, Any]] | None,
    top_k: int,
    taxonomy_tags: set[str],
) -> dict[str, Any]:
    top_ks = (10, 50, top_k)
    retrieval_rows = [
        metric_row([candidate["tag"] for candidate in record["candidates"]], record["gold_tags"], top_ks)
        for record in candidate_records
    ]
    coverage_rows = [bool(record.get("search_coverage")) for record in candidate_records]
    metrics: dict[str, Any] = {
        "n_examples": len(candidate_records),
        "top_k": top_k,
        "gold_taxonomy_coverage": round(
            sum(
                all(tag in taxonomy_tags for tag in record.get("gold_tags", []))
                for record in candidate_records
            )
            / len(candidate_records),
            6,
        )
        if candidate_records
        else 0.0,
        "bm25_retrieval": aggregate_metric_rows(retrieval_rows, top_ks),
        "search_coverage": round(sum(coverage_rows) / len(coverage_rows), 6) if coverage_rows else 0.0,
    }
    metrics["bm25_retrieval"]["search_coverage"] = metrics["search_coverage"]

    if rerank_predictions is not None:
        allowed_examples = {int(record["example_idx"]) for record in candidate_records}
        scoped_predictions = [
            prediction
            for prediction in rerank_predictions
            if int(prediction.get("example_idx", -1)) in allowed_examples
        ]
        rerank_rows = [
            metric_row(prediction.get("final_ranking", []), prediction.get("gold_tags", []), top_ks)
            for prediction in scoped_predictions
        ]
        parse_rows = [bool(prediction.get("parse_ok")) for prediction in scoped_predictions]
        metrics["qwen_reranked"] = aggregate_metric_rows(rerank_rows, top_ks)
        metrics["qwen_reranked"]["parse_success_rate"] = (
            round(sum(parse_rows) / len(parse_rows), 6) if parse_rows else 0.0
        )
        metrics["qwen_reranked"]["search_coverage"] = metrics["search_coverage"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument(
        "--query-mode",
        choices=sorted(QUERY_MODE_ALIASES),
        default="direct_retrieval",
        help=(
            "Retrieval method. Public names are direct_retrieval and "
            "one_pass_grounding; direct and llm_description are accepted aliases."
        ),
    )
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="Reuse output-dir/bm25_candidates.jsonl if it already exists.",
    )
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-rerank", action="store_true", help="Run Qwen reranking after retrieval.")
    parser.add_argument("--rerank-model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--rerank-backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Generation backend for reranking. vLLM batches prompts and is the default.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--query-generation-backend",
        choices=["vllm", "transformers"],
        default="vllm",
    )
    parser.add_argument("--query-description-path", type=Path, default=None)
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    parser.add_argument("--query-max-new-tokens", type=int, default=128)
    parser.add_argument("--query-temperature", type=float, default=0.0)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--context-max-chars", type=int, default=12000)
    parser.add_argument("--candidate-doc-max-chars", type=int, default=320)
    parser.add_argument("--rerank-list-size", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.query_mode = canonical_query_mode(args.query_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = args.output_dir / "bm25_candidates.jsonl"
    query_description_path = args.query_description_path or (args.output_dir / "query_descriptions.jsonl")
    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    query_description_records: list[dict[str, Any]] | None = None
    shared_tokenizer = None
    shared_llm = None

    if args.reuse_candidates and candidates_path.exists():
        candidate_records = load_jsonl(candidates_path)
        if args.limit is not None:
            candidate_records = candidate_records[: args.limit]
        validate_candidate_records(candidate_records, args.query_mode, args.top_k)
        print(f"Reused BM25 candidates from: {candidates_path}")
    else:
        examples = load_examples(args.test_jsonl, limit=args.limit)
        query_descriptions: dict[int, str] | None = None
        if args.query_mode == "one_pass_grounding":
            can_share_vllm = (
                args.run_rerank
                and args.query_generation_backend == "vllm"
                and args.rerank_backend == "vllm"
                and args.query_generation_model == args.rerank_model
            )
            existing_query_records = (
                load_existing_query_descriptions(query_description_path)
                if args.resume and query_description_path.exists()
                else {}
            )
            missing_query_count = sum(
                1 for example in examples if example.example_idx not in existing_query_records
            )
            if can_share_vllm and missing_query_count:
                shared_tokenizer, shared_llm = load_vllm_engine(args, args.query_generation_model)
            query_description_records = generate_query_descriptions(
                args,
                examples,
                query_description_path,
                tokenizer=shared_tokenizer,
                llm=shared_llm,
            )
            query_descriptions = {
                int(record["example_idx"]): normalize_space(record.get("query_description"))
                for record in query_description_records
            }

        candidate_records = build_candidate_records(
            examples,
            taxonomy,
            top_k=args.top_k,
            type_filter=args.type_filter,
            query_mode=args.query_mode,
            query_descriptions=query_descriptions,
        )
        write_jsonl(candidates_path, candidate_records)

    rerank_predictions = None
    rerank_path = args.output_dir / "qwen_rerank_predictions.jsonl"
    if args.run_rerank:
        rerank_predictions = rerank_records(
            args,
            candidate_records,
            rerank_path,
            tokenizer=shared_tokenizer,
            llm=shared_llm,
        )

    if shared_llm is not None or shared_tokenizer is not None:
        release_model_handles(shared_llm, shared_tokenizer)
        shared_llm = None
        shared_tokenizer = None

    if query_description_records is None and args.query_mode == "one_pass_grounding" and query_description_path.exists():
        query_description_records = list(load_existing_query_descriptions(query_description_path).values())
        if args.limit is not None:
            allowed = {int(record["example_idx"]) for record in candidate_records}
            query_description_records = [
                record for record in query_description_records if int(record["example_idx"]) in allowed
            ]

    metrics = evaluate_records(
        candidate_records,
        rerank_predictions,
        top_k=args.top_k,
        taxonomy_tags={concept.tag for concept in taxonomy},
    )
    metrics.update(
        {
            "test_jsonl": str(args.test_jsonl),
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "candidate_path": str(candidates_path),
            "query_mode": args.query_mode,
            "query_description_path": str(query_description_path) if args.query_mode == "one_pass_grounding" else None,
            "query_generation_model": args.query_generation_model if args.query_mode == "one_pass_grounding" else None,
            "query_generation_backend": args.query_generation_backend if args.query_mode == "one_pass_grounding" else None,
            "query_generation_parse_success_rate": (
                round(
                    sum(bool(record.get("parse_ok")) for record in query_description_records)
                    / len(query_description_records),
                    6,
                )
                if query_description_records
                else None
            ),
            "rerank_prediction_path": str(rerank_path) if args.run_rerank else None,
            "type_filter": args.type_filter,
            "rerank_model": args.rerank_model if args.run_rerank else None,
            "rerank_backend": args.rerank_backend if args.run_rerank else None,
        }
    )
    metrics_path = args.output_dir / ("metrics.json" if args.run_rerank else "bm25_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
