import argparse
import os

import torch

from src.model import ILM, TinyConfig


def main():
    parser = argparse.ArgumentParser(description="Dynamic quantization for CPU-friendly ILM inference")
    parser.add_argument("--model", required=True, help="Input fp32 checkpoint path")
    parser.add_argument("--out", default="models/tiny_int8.pt", help="Output quantized checkpoint path")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    cfg = TinyConfig(**ckpt["config"])
    model = ILM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "model_state": quantized.state_dict(),
        "config": ckpt["config"],
        "quantized": True,
    }
    if "tokenizer" in ckpt:
        payload["tokenizer"] = ckpt["tokenizer"]
    if "tokenizer_model_b64" in ckpt:
        payload["tokenizer_model_b64"] = ckpt["tokenizer_model_b64"]
    if "stoi" in ckpt:
        payload["stoi"] = ckpt["stoi"]
    if "itos" in ckpt:
        payload["itos"] = ckpt["itos"]
    torch.save(payload, args.out)
    print(f"saved quantized model: {args.out}")


if __name__ == "__main__":
    main()
