"""Merge a trained LoRA adapter into its base model so vLLM can serve it directly.

python -m training.merge_adapter --base google/gemma-2b \
    --adapter /results/training/<run>/adapter --output /results/models/<run>
"""

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    print(f"merged model written to {args.output}")


if __name__ == "__main__":
    main()
