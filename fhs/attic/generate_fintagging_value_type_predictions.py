#!/usr/bin/env python3
"""Generate predictions for the FinTagging value/type SFT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name/path, usually merged SFT model.")
    parser.add_argument(
        "--dataset-path",
        default="FinTagging_800_200_value_type_sft_arrow",
        help="Arrow DatasetDict path.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    ds = load_from_disk(args.dataset_path)[args.split]
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    do_sample = args.temperature > 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row in ds:
            prompt = row["query"].rstrip() + "\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            handle.write(
                json.dumps(
                    {
                        "source_sample_idx": row["source_sample_idx"],
                        "context_id": row["context_id"],
                        "prediction": prediction,
                        "gold": row["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()
