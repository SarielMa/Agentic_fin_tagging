#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model for value/type extraction inference."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model name/path.")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory.")
    parser.add_argument("--out", required=True, help="Output directory for merged model.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print(f"Merged model saved to: {args.out}")


if __name__ == "__main__":
    main()
