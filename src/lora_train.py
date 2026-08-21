import argparse
import json
import math
import os
from typing import Any

import torch

from src.device_bridge import detect_accelerator
from src.federated.lora import (
    LoRAConfig,
    add_states,
    get_lora_state_dict,
    inject_lora,
    load_lora_state_dict,
    zero_lora_state_like,
)
from src.ilm_profile import apply_cpu_thread_budget, detect_resources, resolve_resource_config, resolve_model_config, ResourceConfig
from src.model import TinyConfig, ILM
from src.resource_defaults import add_resource_pct_args
from src.tokenizer import tokenizer_from_checkpoint, sanitize_text_for_tokenizer
from src.train import (
    _save_bundle,
    build_loss_mask,
    estimate_loss,
    get_batch,
    load_text,
)
from src.training_config import TrainingConfig, build_training_optimizer, compute_lr, apply_embedding_norm
from src.data_curriculum import curriculum_sort_file


def main():
    parser = argparse.ArgumentParser(description="LoRA adapter fine-tuning for Ickle models")
    parser.add_argument("--base-model", required=True, help="Pretrained base model checkpoint")
    parser.add_argument("--data", required=True, help="Instruction fine-tuning data")
    parser.add_argument("--out", default="models/tiny_lora.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--contrastive-coeff", type=float, default=0.0)
    parser.add_argument("--embed-norm", action="store_true")
    parser.add_argument("--curriculum-sort", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--targets", default="q,k,v,proj,w1,w2,w3,lm_head")
    parser.add_argument("--seed", type=int, default=1337)
    add_resource_pct_args(parser)
    parser.add_argument("--block-size", type=int, default=0, help="Override model block size (0 = auto)")
    parser.add_argument("--n-embd", type=int, default=0, help="Override model embedding dim (0 = auto)")
    parser.add_argument("--n-head", type=int, default=0, help="Override model attention heads (0 = auto)")
    parser.add_argument("--n-layer", type=int, default=0, help="Override model layers (0 = auto)")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--resume-lora", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tc = TrainingConfig.from_args(args)
    tc.lr = args.lr
    tc.steps = args.steps
    tc.batch_size = args.batch_size
    tc.grad_accum_steps = args.grad_accum_steps
    tc.weight_decay = args.weight_decay
    tc.grad_clip = args.grad_clip
    tc.warmup_steps = args.warmup_steps
    tc.eval_every = args.eval_every
    tc.eval_iters = args.eval_iters

    rc = resolve_resource_config(args)
    torch_threads = args.torch_threads if args.torch_threads > 0 else rc.torch_threads
    apply_cpu_thread_budget(torch_threads)
    accel = detect_accelerator()
    device = accel.device

    base_path = args.base_model
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base model not found: {base_path}")
    bundle = torch.load(base_path, map_location=device)
    cfg_dict = bundle.get("config") or {}
    cfg = TinyConfig(**cfg_dict)
    tokenizer = tokenizer_from_checkpoint(bundle)

    text = load_text(args.data)
    text = sanitize_text_for_tokenizer(text, tokenizer)
    if tc.curriculum_sort:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
            f.write(text)
            tmp_path = f.name
        sorted_path = tmp_path + ".curriculum.txt"
        curriculum_sort_file(tmp_path, sorted_path, ascending=True)
        with open(sorted_path, "r", encoding="utf-8") as f:
            text = f.read()
        os.unlink(tmp_path)
        try:
            os.unlink(sorted_path)
        except OSError:
            pass
        print("Curriculum sorting applied")
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = ILM(cfg).to(device)
    model.load_state_dict(bundle["model_state"])
    if tc.contrastive_coeff > 0:
        model.cfg.contrastive_coeff = tc.contrastive_coeff

    lora_targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    lora_cfg = LoRAConfig(rank=args.rank, alpha=args.alpha, dropout=args.dropout, target_modules=lora_targets)
    inject_lora(model, lora_cfg)
    print(f"LoRA injected: rank={args.rank}, alpha={args.alpha}")

    if args.resume_lora:
        resume_path = args.resume_lora
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Lora checkpoint not found: {resume_path}")
        resume_bundle = torch.load(resume_path, map_location=device)
        if "lora_state" in resume_bundle:
            load_lora_state_dict(model, resume_bundle["lora_state"])
        elif "model_state" in resume_bundle:
            load_lora_state_dict(model, resume_bundle["model_state"])
        else:
            load_lora_state_dict(model, resume_bundle)
        print(f"Resumed LoRA from: {resume_path}")

    effective_batch = tc.batch_size if tc.batch_size > 0 else 4
    grad_accum_steps = max(1, tc.grad_accum_steps)

    n = int(0.9 * len(encoded))
    train_data = encoded[:n]
    val_data = encoded[n:]
    loss_mask = build_loss_mask(encoded.tolist(), tokenizer, text)
    train_mask = loss_mask[:n]
    val_mask = loss_mask[n:]

    optimizer = build_training_optimizer(model, tc)
    print(f"Optimizer: {tc.optimizer}, schedule: {tc.lr_schedule}, contrastive: {tc.contrastive_coeff}, embed_norm: {tc.embed_norm}")

    checkpoint_every = max(0, int(args.checkpoint_every))
    checkpoint_path = args.checkpoint_path.strip() or f"{args.out}.checkpoint.pt"

    def save_checkpoint(step: int, reason: str):
        if not checkpoint_path:
            return
        _save_bundle(
            checkpoint_path,
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=step,
            include_optimizer=True,
            lora_state=get_lora_state_dict(model),
        )

    model.train()
    try:
        for step in range(tc.steps):
            curr_lr = compute_lr(tc, step, tc.steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(grad_accum_steps):
                xb, yb, mb = get_batch(train_data, train_mask, cfg.block_size, effective_batch, device)
                _, micro_loss = model(xb, yb, loss_mask=mb)
                (micro_loss / grad_accum_steps).backward()
                micro_losses.append(float(micro_loss.item()))

            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()

            if tc.embed_norm:
                apply_embedding_norm(model)

            step_train_loss = sum(micro_losses) / max(1, len(micro_losses))

            if step % tc.eval_every == 0 or step == tc.steps - 1:
                val_loss = estimate_loss(model, val_data, val_mask, cfg, effective_batch, device, args.eval_iters)
                print(f"step={step} train_loss={step_train_loss:.4f} val_loss={val_loss:.4f} lr={curr_lr:.6f}")

            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                save_checkpoint(step, reason="interval")
    except KeyboardInterrupt:
        save_checkpoint(step, reason="interrupt")
        print("training interrupted by user")
        raise SystemExit(130)

    _save_bundle(
        args.out,
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        lora_state=get_lora_state_dict(model),
    )

    meta_path = args.out + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": base_path,
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "steps": args.steps,
            "batch_size": effective_batch,
            "learning_rate": args.lr,
        }, f, indent=2)

    print(f"saved LoRA model: {args.out}")


if __name__ == "__main__":
    main()
