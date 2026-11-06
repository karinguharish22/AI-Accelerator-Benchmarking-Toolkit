"""Rough FLOP accounting used for model FLOPs utilization (MFU).

Full fine-tuning costs ~6N FLOPs per token (2N forward, 4N backward). With LoRA
the frozen base weights need no weight gradients, only activation gradients, so
the base model costs ~4N and only the small adapter pays the full 6N. Attention
score FLOPs are ignored, which slightly understates MFU at long sequence lengths.
"""


def train_flops_per_token(total_params: int, trainable_params: int) -> float:
    frozen = total_params - trainable_params
    return 4.0 * frozen + 6.0 * trainable_params


def inference_flops_per_token(total_params: int) -> float:
    return 2.0 * total_params


def mfu(tokens_per_s_per_chip: float, flops_per_token: float, peak_tflops_per_chip: float) -> float:
    if peak_tflops_per_chip <= 0:
        return 0.0
    return tokens_per_s_per_chip * flops_per_token / (peak_tflops_per_chip * 1e12)
