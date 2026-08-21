from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model import ILM


@torch.no_grad()
def speculative_generate(
    target_model: ILM,
    draft_model: ILM,
    idx: torch.Tensor,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    gamma: int = 4,
) -> torch.Tensor:
    """Speculative sampling: draft model proposes tokens, target model verifies.

    Algorithm (Chen et al., 2023):
    1. Draft model generates up to gamma candidate tokens greedily.
    2. Target model scores all gamma+1 positions in one forward pass.
    3. Modified rejection sampling accepts/rejects each draft token,
       preserving the target distribution exactly.

    Args:
        target_model: The full ILM used for verification.
        draft_model: A smaller/faster model used for drafting (same vocab).
        idx: Input token indices of shape (1, seq_len).
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering for the target model.
        gamma: Number of draft tokens to propose per step.

    Returns:
        Tensor of shape (1, seq_len + N) where N <= max_new_tokens.
    """
    if temperature <= 0:
        temperature = 1.0
    device = idx.device
    block_size = target_model.cfg.block_size
    start_len = idx.size(1)
    max_len = start_len + max_new_tokens

    while idx.size(1) < max_len:
        remaining = max_len - idx.size(1)
        actual_gamma = min(gamma, remaining - 1)
        if actual_gamma <= 0:
            draft_logits, _ = target_model(idx[:, -block_size:])
            draft_logits = draft_logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(draft_logits, min(top_k, draft_logits.size(-1)))
                draft_logits = draft_logits.masked_fill(draft_logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(draft_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            continue

        draft_input = idx.clone()
        for _ in range(actual_gamma):
            dblock = draft_input[:, -block_size:]
            dlogits, _ = draft_model(dblock)
            dnext = torch.argmax(dlogits[:, -1, :], dim=-1, keepdim=True)
            draft_input = torch.cat((draft_input, dnext), dim=1)

        draft_candidates = draft_input[:, -actual_gamma:]
        target_input = torch.cat([idx[:, -block_size:], draft_candidates], dim=1)
        target_logits, _ = target_model(target_input)
        target_logits = target_logits[:, -(actual_gamma + 1):, :] / temperature

        all_accepted = True
        for t in range(actual_gamma):
            p = F.softmax(target_logits[:, t, :], dim=-1)
            if top_k > 0:
                v, _ = torch.topk(p, min(top_k, p.size(-1)))
                p = p.masked_fill(p < v[:, [-1]], 0.0)
                p = p / p.sum(dim=-1, keepdim=True)

            draft_token = draft_candidates[0, t].item()
            q = torch.zeros_like(p)
            q[0, draft_token] = 1.0

            p_clamped = torch.maximum(p, q * 1e-6)
            p_ratio = p_clamped / torch.clamp(q, min=1e-6)
            r = torch.rand(1, device=device)
            if r.item() < p_ratio[0, draft_token].item():
                idx = torch.cat((idx, draft_candidates[:, t:t + 1]), dim=1)
            else:
                all_accepted = False
                residual = torch.clamp(p - q, min=0)
                residual_sum = residual.sum()
                if residual_sum > 1e-8:
                    residual = residual / residual_sum
                    idx_next = torch.multinomial(residual, num_samples=1)
                else:
                    idx_next = torch.multinomial(p, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                break

            if idx.size(1) >= max_len:
                break

        if all_accepted and idx.size(1) < max_len:
            p_bonus = F.softmax(target_logits[:, actual_gamma, :], dim=-1)
            if top_k > 0:
                v, _ = torch.topk(p_bonus, min(top_k, p_bonus.size(-1)))
                p_bonus = p_bonus.masked_fill(p_bonus < v[:, [-1]], 0.0)
                p_bonus = p_bonus / p_bonus.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(p_bonus, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        if idx.size(1) >= max_len:
            break

    return idx


@torch.no_grad()
def speculative_generate_simple(
    model: ILM,
    idx: torch.Tensor,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    gamma: int = 3,
) -> torch.Tensor:
    """Simplified speculative decoding: greedy-draft + sampling-verify on same model.

    The model does two forward passes per step:
    1. Greedy draft: generates gamma tokens with argmax.
    2. Verify: scores all gamma tokens in one pass with full sampling.

    On GPU, batch-scoring gamma tokens in one pass is faster
    than sampling gamma tokens one-by-one.

    Returns:
        Tensor of shape (1, seq_len + N).
    """
    if temperature <= 0:
        temperature = 1.0
    device = idx.device
    block_size = model.cfg.block_size
    start_len = idx.size(1)
    max_len = start_len + max_new_tokens

    while idx.size(1) < max_len:
        remaining = max_len - idx.size(1)
        actual_gamma = min(gamma, remaining - 1)
        if actual_gamma <= 0:
            cond = idx[:, -block_size:]
            logits, _ = model(cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            continue

        cond = idx[:, -block_size:]
        draft_input = cond.clone()

        for _ in range(actual_gamma):
            dblock = draft_input[:, -block_size:]
            dlogits, _ = model(dblock)
            dnext = torch.argmax(dlogits[:, -1, :], dim=-1, keepdim=True)
            draft_input = torch.cat((draft_input, dnext), dim=1)

        candidates = draft_input[:, -actual_gamma:]
        target_input = torch.cat([cond, candidates], dim=1)
        logits_all, _ = model(target_input)
        logits_all = logits_all[:, -(actual_gamma + 1):, :] / temperature

        all_accepted = True
        for t in range(actual_gamma):
            p = F.softmax(logits_all[:, t, :], dim=-1)
            if top_k > 0:
                v, _ = torch.topk(p, min(top_k, p.size(-1)))
                p = p.masked_fill(p < v[:, [-1]], 0.0)
                p = p / p.sum(dim=-1, keepdim=True)

            draft_token = candidates[0, t].item()
            q = torch.zeros_like(p)
            q[0, draft_token] = 1.0

            p_clamped = torch.maximum(p, q * 1e-6)
            p_ratio = p_clamped / torch.clamp(q, min=1e-6)
            r = torch.rand(1, device=device)
            if r.item() < p_ratio[0, draft_token].item():
                idx = torch.cat((idx, candidates[:, t:t + 1]), dim=1)
            else:
                all_accepted = False
                residual = torch.clamp(p - q, min=0)
                residual_sum = residual.sum()
                if residual_sum > 1e-8:
                    residual = residual / residual_sum
                    idx_next = torch.multinomial(residual, num_samples=1)
                else:
                    idx_next = torch.multinomial(p, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                break

            if idx.size(1) >= max_len:
                break

        if all_accepted and idx.size(1) < max_len:
            p_bonus = F.softmax(logits_all[:, actual_gamma, :], dim=-1)
            if top_k > 0:
                v, _ = torch.topk(p_bonus, min(top_k, p_bonus.size(-1)))
                p_bonus = p_bonus.masked_fill(p_bonus < v[:, [-1]], 0.0)
                p_bonus = p_bonus / p_bonus.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(p_bonus, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx
