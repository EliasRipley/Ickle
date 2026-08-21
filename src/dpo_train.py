from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.device_bridge import detect_accelerator
from src.ilm_profile import apply_cpu_thread_budget
from src.model import ILM, TinyConfig
from src.tokenizer import BaseTokenizer, tokenizer_from_checkpoint
from src.training_config import TrainingConfig, build_training_optimizer, apply_embedding_norm


@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def _load_model_bundle(model_path: str, device: str) -> tuple[ILM, BaseTokenizer, dict[str, Any]]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = TinyConfig(**ckpt["config"])
    tokenizer = tokenizer_from_checkpoint(ckpt)

    model = ILM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, tokenizer, ckpt


def _clip_pair(prompt_ids: list[int], completion_ids: list[int], max_total: int, fallback_prompt_id: int = 0) -> tuple[list[int], int]:
    if not completion_ids:
        completion_ids = [fallback_prompt_id]
    if not prompt_ids:
        prompt_ids = [fallback_prompt_id]

    # Prefer to trim the prompt, not the completion, to preserve the response signal.
    needed = 1 + len(completion_ids)  # at least 1 prompt token
    if needed > max_total:
        completion_ids = completion_ids[: max_total - 1]
        needed = 1 + len(completion_ids)
    if needed + len(prompt_ids) > max_total:
        keep = max(1, max_total - len(completion_ids))
        prompt_ids = prompt_ids[-keep:]

    full = prompt_ids + completion_ids
    return full, len(prompt_ids)


def _completion_logprob_sum(
    model: ILM,
    *,
    prompt: str,
    completion: str,
    tokenizer: BaseTokenizer,
) -> torch.Tensor:
    prompt_ids = [int(i) for i in tokenizer.encode(prompt)]
    completion_ids = [int(i) for i in tokenizer.encode(completion)]
    max_total = int(model.cfg.block_size) + 1
    fallback = 0
    full_ids, prompt_len = _clip_pair(prompt_ids, completion_ids, max_total=max_total, fallback_prompt_id=fallback)

    if len(full_ids) < 2:
        return torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)

    idx = torch.tensor(full_ids[:-1], dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    targets = torch.tensor(full_ids[1:], dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    logits, _ = model(idx, targets)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_targets = targets[0]
    positions = torch.arange(token_targets.numel(), device=token_targets.device)
    token_logprobs = log_probs[0, positions, token_targets]

    completion_len = len(full_ids) - prompt_len
    start = max(0, prompt_len - 1)
    end = min(token_logprobs.numel(), start + completion_len)
    if end <= start:
        return token_logprobs.new_tensor(0.0)
    return token_logprobs[start:end].sum()


def _read_preference_data(path: str) -> list[PreferenceSample]:
    out: list[PreferenceSample] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"preference data not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            prompt = str(item.get("prompt", "")).strip()
            chosen = str(item.get("chosen", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected:
                continue
            if chosen == rejected:
                continue
            out.append(PreferenceSample(prompt=prompt, chosen=chosen, rejected=rejected))
    if not out:
        raise ValueError("No valid preference rows found.")
    return out


def _dpo_loss(
    policy_model: ILM,
    ref_model: ILM,
    *,
    sample: PreferenceSample,
    tokenizer: BaseTokenizer,
    beta: float,
) -> torch.Tensor:
    pi_chosen = _completion_logprob_sum(policy_model, prompt=sample.prompt, completion=sample.chosen, tokenizer=tokenizer)
    pi_rejected = _completion_logprob_sum(
        policy_model,
        prompt=sample.prompt,
        completion=sample.rejected,
        tokenizer=tokenizer,
    )

    with torch.no_grad():
        ref_chosen = _completion_logprob_sum(ref_model, prompt=sample.prompt, completion=sample.chosen, tokenizer=tokenizer)
        ref_rejected = _completion_logprob_sum(
            ref_model,
            prompt=sample.prompt,
            completion=sample.rejected,
            tokenizer=tokenizer,
        )

    logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(logits)


def _save_model(
    *,
    out_path: str,
    policy_model: ILM,
    cfg: TinyConfig,
    tokenizer: BaseTokenizer,
):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": policy_model.state_dict(),
        "config": vars(cfg),
        "trained_with": "dpo",
    }
    payload.update(tokenizer.checkpoint_payload())
    torch.save(payload, str(out))


def _compute_preference_accuracy(
    policy_model: ILM,
    eval_samples: list[PreferenceSample],
    tokenizer: BaseTokenizer,
) -> float:
    """Fraction of eval pairs where chosen logprob > rejected logprob."""
    if not eval_samples:
        return 0.0
    correct = 0
    for sample in eval_samples:
        pi_chosen = _completion_logprob_sum(policy_model, prompt=sample.prompt, completion=sample.chosen, tokenizer=tokenizer)
        pi_rejected = _completion_logprob_sum(policy_model, prompt=sample.prompt, completion=sample.rejected, tokenizer=tokenizer)
        if float(pi_chosen) > float(pi_rejected):
            correct += 1
    return correct / len(eval_samples)


def run_dpo(
    *,
    model_path: str,
    preference_data_path: str,
    out_path: str,
    reference_model_path: str = "",
    steps: int = 300,
    batch_size: int = 2,
    lr: float = 1e-5,
    beta: float = 0.1,
    grad_clip: float = 1.0,
    optimizer: str = "adamw",
    embed_norm: bool = False,
    seed: int = 1337,
    torch_threads: int = 4,
    eval_every: int = 25,
    val_split: float = 0.1,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    apply_cpu_thread_budget(torch_threads)
    accel = detect_accelerator()
    device = accel.device

    policy_model, tokenizer, ckpt = _load_model_bundle(model_path, device=device)
    ref_path = reference_model_path.strip() or model_path
    ref_model, ref_tokenizer, _ = _load_model_bundle(ref_path, device=device)
    if tokenizer.kind != ref_tokenizer.kind or tokenizer.vocab_size != ref_tokenizer.vocab_size:
        raise ValueError("Reference model tokenizer does not match policy model tokenizer.")

    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    policy_model.train()

    tc = TrainingConfig(lr=lr, optimizer=optimizer, embed_norm=embed_norm, weight_decay=0.0)
    optimizer = build_training_optimizer(policy_model, tc)
    samples = _read_preference_data(preference_data_path)
    random.shuffle(samples)
    split = max(1, int(len(samples) * (1.0 - val_split)))
    train_samples = samples[:split]
    eval_samples = samples[split:]
    sampled_count = len(train_samples)

    initial_acc = _compute_preference_accuracy(policy_model, eval_samples, tokenizer)
    print(f"Baseline preference accuracy: {initial_acc:.2%}")

    history: list[dict[str, float]] = []
    for step in range(max(1, int(steps))):
        batch = random.choices(train_samples, k=max(1, int(batch_size)))
        losses = [
            _dpo_loss(policy_model, ref_model, sample=sample, tokenizer=tokenizer, beta=float(beta))
            for sample in batch
        ]
        loss = torch.stack(losses).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max(0.01, float(grad_clip)))
        optimizer.step()
        if embed_norm:
            apply_embedding_norm(policy_model)

        if (step % max(1, int(eval_every)) == 0) or (step == int(steps) - 1):
            val_acc = _compute_preference_accuracy(policy_model, eval_samples, tokenizer)
            history.append({"step": float(step), "loss": float(loss.item()), "val_accuracy": val_acc})
            print(f"dpo step={step} loss={loss.item():.4f} val_acc={val_acc:.2%}")

    _save_model(
        out_path=out_path,
        policy_model=policy_model,
        cfg=TinyConfig(**ckpt["config"]),
        tokenizer=tokenizer,
    )

    meta = {
        "method": "dpo",
        "model_path": model_path,
        "reference_model_path": ref_path,
        "preference_data_path": preference_data_path,
        "out_path": out_path,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "beta": float(beta),
        "grad_clip": float(grad_clip),
        "seed": int(seed),
        "torch_threads": int(torch_threads),
        "sample_count": sampled_count,
        "initial_accuracy": initial_acc,
        "history": history,
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) training for Ickle.")
    parser.add_argument("--model", required=True, help="Input policy model checkpoint (.pt)")
    parser.add_argument("--prefs", required=True, help="Preference JSONL with prompt/chosen/rejected")
    parser.add_argument("--out", default="models/ickle_dpo.pt")
    parser.add_argument("--reference-model", default="", help="Optional fixed reference model (defaults to --model)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--embed-norm", action="store_true", help="Normalize embeddings after each step")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_dpo(
        model_path=args.model,
        preference_data_path=args.prefs,
        out_path=args.out,
        reference_model_path=args.reference_model,
        steps=max(1, int(args.steps)),
        batch_size=max(1, int(args.batch_size)),
        lr=max(1e-7, float(args.lr)),
        beta=max(1e-6, float(args.beta)),
        grad_clip=max(0.01, float(args.grad_clip)),
        optimizer=args.optimizer,
        embed_norm=bool(args.embed_norm),
        seed=int(args.seed),
        torch_threads=max(1, int(args.torch_threads)),
        eval_every=max(1, int(args.eval_every)),
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"saved DPO model: {args.out}")
        print(f"sample_count: {report['sample_count']}")
        if report.get("history"):
            print(f"final_loss: {report['history'][-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
