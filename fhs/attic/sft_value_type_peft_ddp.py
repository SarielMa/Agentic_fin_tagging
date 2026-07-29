#!/usr/bin/env python3
"""DDP PEFT/QLoRA SFT for the FinTagging value/type extraction task.

This mirrors the PV Miner training setup:
- dataset is loaded with datasets.load_from_disk()
- each example has query and answer columns
- training text is query + "\n" + answer
- labels are masked so loss is applied only to answer tokens
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def build_features(example: dict[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    query = example["query"].rstrip()
    answer = example["answer"].strip()

    prompt_text = query + "\n"
    full_text = prompt_text + answer

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = [-100] * len(input_ids)

    answer_start = min(len(prompt_ids), len(input_ids))
    for idx in range(answer_start, len(input_ids)):
        labels[idx] = input_ids[idx]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class LabelPaddingCollator(DataCollatorWithPadding):
    """Pad labels with -100 while the tokenizer pads input_ids/attention_mask."""

    def __call__(self, features):
        batch = super().__call__([{k: v for k, v in item.items() if k != "labels"} for item in features])
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for item in features:
            labels = item["labels"]
            padded_labels.append(labels + [-100] * (max_len - len(labels)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="./sft_value_type_out")
    parser.add_argument("--max_length", type=int, default=8192)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--do_eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    if hasattr(dataset, "keys") and "train" in dataset:
        train_ds = dataset["train"]
        eval_ds = dataset.get("test")
    else:
        train_ds = dataset
        eval_ds = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quant_config = None
    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map=None,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        return build_features(example, tokenizer=tokenizer, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names, desc="Tokenizing train")
    eval_tok = None
    if args.do_eval and eval_ds is not None:
        eval_tok = eval_ds.map(tokenize, remove_columns=eval_ds.column_names, desc="Tokenizing eval")
        if len(eval_tok) == 0:
            eval_tok = None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        fp16=not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=LabelPaddingCollator(tokenizer=tokenizer, padding=True),
        tokenizer=tokenizer,
    )
    trainer.train()

    if rank == 0:
        adapter_dir = output_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"\nSaved LoRA adapter to: {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
