import pytest

from training.flops import inference_flops_per_token, mfu, train_flops_per_token


def test_lora_flops_between_inference_and_full_finetune():
    n = 2_000_000_000
    lora = train_flops_per_token(n, trainable_params=4_000_000)
    assert inference_flops_per_token(n) < lora < 6 * n
    assert train_flops_per_token(n, n) == 6 * n


def test_mfu():
    # 1000 tok/s * 1e11 FLOPs/token = 1e14 FLOP/s = 100 TFLOP/s of a 200 TFLOP/s chip.
    assert mfu(1000, 1e11, 200) == pytest.approx(0.5)
    assert mfu(1000, 1e11, 0) == 0.0
