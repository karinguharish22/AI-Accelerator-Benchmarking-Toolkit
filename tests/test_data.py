import pytest

from training.data import format_example, padded_length


def test_format_example_with_and_without_context():
    with_ctx = format_example({"instruction": " Summarize ", "context": "Some text.", "response": "Short."})
    assert "### Context:\nSome text." in with_ctx
    assert with_ctx.startswith("### Instruction:\nSummarize")
    assert with_ctx.endswith("### Response:\nShort.")

    no_ctx = format_example({"instruction": "Hi", "context": "", "response": "Hello"})
    assert "### Context" not in no_ctx


@pytest.mark.parametrize(
    "mode, lengths, expected",
    [
        ("max_length", [10, 50], 1024),
        ("longest", [10, 50], 50),
        ("bucket", [10, 50], 128),
        ("bucket", [129], 256),
        ("bucket", [1020], 1024),  # never exceeds seq_len
    ],
)
def test_padded_length(mode, lengths, expected):
    assert padded_length(lengths, mode, seq_len=1024, bucket_size=128) == expected


def test_padded_length_rejects_unknown_mode():
    with pytest.raises(ValueError):
        padded_length([1], "weird", 8)


def test_collator_masks_padding_from_loss():
    pytest.importorskip("torch")
    from training.data import make_collator

    collate = make_collator(pad_token_id=0, mode="longest", seq_len=16)
    batch = collate([{"input_ids": [5, 6, 7]}, {"input_ids": [8]}])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[5, 6, 7], [8, -100, -100]]


def test_collator_static_shapes():
    pytest.importorskip("torch")
    from training.data import make_collator

    collate = make_collator(pad_token_id=0, mode="max_length", seq_len=8)
    assert collate([{"input_ids": [1, 2]}])["input_ids"].shape == (1, 8)
