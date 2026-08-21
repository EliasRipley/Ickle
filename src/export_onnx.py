import argparse
import os

import torch

from src.model import TinyConfig, ILM


def main():
    parser = argparse.ArgumentParser(description="Export ILM checkpoint to ONNX for portable runtimes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="models/tiny.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    cfg = TinyConfig(**ckpt["config"])
    model = ILM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros((1, min(16, cfg.block_size)), dtype=torch.long)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ILM.forward(idx) with no `targets` -- the only call shape that makes
    # sense for an inference-only export -- returns (logits, None). Exporting
    # with output_names=["logits", "loss"] declared a second output that was
    # never actually a tensor, which torch.onnx.export cannot trace; a
    # portable-runtime consumer has no targets to compute a loss against
    # anyway, so logits is the only output that means anything here.
    #
    # dynamo=False pins this to torch's TorchScript-based exporter: recent
    # torch defaults to the dynamo-based one, which imports `onnxscript` --
    # an extra dependency this project doesn't otherwise need. The legacy
    # exporter still needs the `onnx` package itself (see
    # requirements-onnx.txt) to write the .onnx file, which this command
    # had no requirements file for at all before.
    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported ONNX: {args.out}")


if __name__ == "__main__":
    main()
