"""Instruction-tuning data pipeline.

Padding strategy matters a lot on TPUs: XLA compiles one graph per input shape,
so `longest` padding (a new shape almost every batch) triggers constant
recompilation, while `max_length` (one shape) compiles once. `bucket` is the
middle ground: pad up to the next multiple of `bucket_size`, so there are only
seq_len / bucket_size possible shapes.
"""

from __future__ import annotations

import math

PADDING_MODES = ("max_length", "bucket", "longest")

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n{context_block}### Response:\n{response}"


def format_example(example: dict) -> str:
    """Render one Dolly-style record (instruction / context / response) as training text."""
    context = (example.get("context") or "").strip()
    context_block = f"### Context:\n{context}\n\n" if context else ""
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"].strip(),
        context_block=context_block,
        response=example["response"].strip(),
    )


def padded_length(lengths: list[int], mode: str, seq_len: int, bucket_size: int = 128) -> int:
    """Length every sequence in a batch is padded to, for the given padding mode."""
    if mode == "max_length":
        return seq_len
    longest = max(lengths)
    if mode == "longest":
        return longest
    if mode == "bucket":
        return min(seq_len, math.ceil(longest / bucket_size) * bucket_size)
    raise ValueError(f"unknown padding mode {mode!r}; expected one of {PADDING_MODES}")


def make_collator(pad_token_id: int, mode: str, seq_len: int, bucket_size: int = 128):
    """Build a DataLoader collate_fn that pads token ids and masks padding out of the loss."""
    import torch

    def collate(features: list[dict]) -> dict:
        lengths = [len(f["input_ids"]) for f in features]
        target = padded_length(lengths, mode, seq_len, bucket_size)
        input_ids = torch.full((len(features), target), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), target), dtype=torch.long)
        for row, f in enumerate(features):
            ids = f["input_ids"][:target]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def build_dataset(tokenizer, name: str, seq_len: int, max_examples: int | None = None):
    """Load, format and tokenize the dataset. Padding happens later, in the collator."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    ds = ds.map(lambda ex: {"text": format_example(ex) + tokenizer.eos_token}, remove_columns=ds.column_names)
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=seq_len),
        batched=True,
        remove_columns=["text"],
    )
    return ds.remove_columns([c for c in ds.column_names if c != "input_ids"])
