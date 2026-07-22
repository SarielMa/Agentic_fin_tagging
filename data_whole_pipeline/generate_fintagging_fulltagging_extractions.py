#!/usr/bin/env python3
"""Run FinTagging text/table extractors on the original test split."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_fintagging_context_extraction_sft_datasets import build_query


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ORIGINAL_TEST = SCRIPT_DIR / "FinTagging_800_200_HF" / "data" / "test.parquet"
DEFAULT_TEXT_MODEL = (
    SCRIPT_DIR / "runs_fintagging_text_context" / "qwen2.5_14b_instruct" / "sft_3ep" / "merged"
)
DEFAULT_TABLE_MODEL = (
    SCRIPT_DIR / "runs_fintagging_table_context" / "qwen2.5_14b_instruct" / "sft_3ep" / "merged"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-test-parquet", type=Path, default=DEFAULT_ORIGINAL_TEST)
    parser.add_argument("--text-extractor-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--table-extractor-model", type=Path, default=DEFAULT_TABLE_MODEL)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--task", choices=["all", "text", "table"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text-max-new-tokens", type=int, default=2048)
    parser.add_argument("--table-max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output without using existing rows to skip predictions.",
    )
    return parser.parse_args()


def has_table_markup(text: Any) -> bool:
    return "<table" in str(text).lower()


def routed_task(text: Any) -> str:
    return "table" if has_table_markup(text) else "text"


def load_existing_predictions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["source_sample_idx"])] = row
    return rows


def load_model(model_path: Path, bf16: bool) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def release_model(tokenizer: Any, model: Any) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_vllm_model(args: argparse.Namespace, model_path: Path) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": str(model_path),
        "dtype": "bfloat16" if args.bf16 else "float16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    return LLM(**llm_kwargs)


def release_vllm_model(llm: Any) -> None:
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_prediction(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt.rstrip() + "\n", return_tensors="pt").to(model.device)
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


def generate_predictions_vllm(
    args: argparse.Namespace,
    llm: Any,
    pending: list[dict[str, Any]],
    task: str,
    max_new_tokens: int,
    handle: Any,
    model_path: Path,
) -> None:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(pending), batch_size):
        batch_rows = pending[start : start + batch_size]
        prompts = [build_query(row["text"], task=task).rstrip() + "\n" for row in batch_rows]
        outputs = llm.generate(prompts, sampling_params)
        for row, output in zip(batch_rows, outputs):
            prediction = output.outputs[0].text.strip() if output.outputs else ""
            out = {
                "source_sample_idx": int(row["source_sample_idx"]),
                "context_id": int(row["context_id"]),
                "split": args.split,
                "input_type": task,
                "prediction": prediction,
                "context": row["text"],
                "extractor_model": str(model_path),
                "backend": "vllm",
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
        handle.flush()
        done = min(start + batch_size, len(pending))
        print(f"Generated {done}/{len(pending)} {task} extraction predictions")


def generate_predictions_transformers(
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    pending: list[dict[str, Any]],
    task: str,
    max_new_tokens: int,
    handle: Any,
    model_path: Path,
) -> None:
    for offset, row in enumerate(pending, start=1):
        prompt = build_query(row["text"], task=task)
        prediction = generate_prediction(
            tokenizer,
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        out = {
            "source_sample_idx": int(row["source_sample_idx"]),
            "context_id": int(row["context_id"]),
            "split": args.split,
            "input_type": task,
            "prediction": prediction,
            "context": row["text"],
            "extractor_model": str(model_path),
            "backend": "transformers",
        }
        handle.write(json.dumps(out, ensure_ascii=False) + "\n")
        handle.flush()
        if offset % 10 == 0 or offset == len(pending):
            print(f"Generated {offset}/{len(pending)} {task} extraction predictions")


def dataframe_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    rows = df.to_dict(orient="records")
    for row_idx, row in enumerate(rows):
        row["source_sample_idx"] = int(row.get("source_sample_idx", row_idx))
        row["context_id"] = int(row.get("context_id", row["source_sample_idx"]))
        row["text"] = str(row.get("text", ""))
        row["input_type"] = routed_task(row["text"])
    return rows


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    rows = dataframe_rows(args.original_test_parquet, args.limit)
    existing = load_existing_predictions(args.output_jsonl) if args.resume else {}
    mode = "a" if (args.append or (args.resume and args.output_jsonl.exists())) else "w"
    selected_tasks = ("text", "table") if args.task == "all" else (args.task,)
    pending_by_task = {
        task: [
            row
            for row in rows
            if row["input_type"] == task and int(row["source_sample_idx"]) not in existing
        ]
        for task in selected_tasks
    }

    stats: dict[str, Any] = {
        "original_test_parquet": str(args.original_test_parquet),
        "output_jsonl": str(args.output_jsonl),
        "split": args.split,
        "routing_method": "html_table_markup",
        "selected_tasks": list(selected_tasks),
        "sample_count": len(rows),
        "input_type_counts": {
            task: sum(1 for row in rows if row["input_type"] == task)
            for task in ("text", "table")
        },
        "backend": args.backend,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "existing_prediction_count": len(existing),
        "pending_prediction_count": sum(len(items) for items in pending_by_task.values()),
        "text_extractor_model": str(args.text_extractor_model),
        "table_extractor_model": str(args.table_extractor_model),
    }

    with args.output_jsonl.open(mode, encoding="utf-8") as handle:
        task_configs = {
            "text": (args.text_extractor_model, args.text_max_new_tokens),
            "table": (args.table_extractor_model, args.table_max_new_tokens),
        }
        for task in selected_tasks:
            model_path, max_new_tokens = task_configs[task]
            pending = pending_by_task[task]
            if not pending:
                continue

            if args.backend == "vllm":
                llm = load_vllm_model(args, model_path)
                try:
                    generate_predictions_vllm(
                        args=args,
                        llm=llm,
                        pending=pending,
                        task=task,
                        max_new_tokens=max_new_tokens,
                        handle=handle,
                        model_path=model_path,
                    )
                finally:
                    release_vllm_model(llm)
            else:
                tokenizer, model = load_model(model_path, bf16=args.bf16)
                try:
                    generate_predictions_transformers(
                        args=args,
                        tokenizer=tokenizer,
                        model=model,
                        pending=pending,
                        task=task,
                        max_new_tokens=max_new_tokens,
                        handle=handle,
                        model_path=model_path,
                    )
                finally:
                    release_model(tokenizer, model)

    if args.metadata_json is not None:
        args.metadata_json.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
