import math
import numpy as np
import time


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        for name in params.keys():
            p = params[name]
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def zero_grad(self):
        pass

    def step(self, grads):
        self.t += 1
        for name in self.params.keys():
            if name not in grads:
                continue
            g = grads[name]
            p = self.params[name]
            self.m[name] = self.betas[0] * self.m[name] + (1 - self.betas[0]) * g
            self.v[name] = self.betas[1] * self.v[name] + (1 - self.betas[1]) * (g * g)
            m_hat = self.m[name] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[name] / (1 - self.betas[1] ** self.t)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            update = update + self.weight_decay * p
            self.params[name] = p - self.lr * update


def _get_batch(tokens, block_size, batch_size):
    n = len(tokens) - block_size
    starts = np.random.randint(0, n, size=batch_size)
    x = np.array([tokens[s:s + block_size] for s in starts], dtype=np.int64)
    y = np.array([tokens[s + 1:s + block_size + 1] for s in starts], dtype=np.int64)
    return x, y


def _cross_entropy_vec(logits_flat, targets_flat):
    probs = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    n = probs.shape[0]
    return -np.mean(np.log(probs[np.arange(n), targets_flat] + 1e-12))


def _compute_loss(model, xb, yb):
    bsz, seqlen = xb.shape
    logits, _ = model.forward(xb, targets=yb)
    logits = logits.astype(np.float64)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = yb.reshape(-1)
    return _cross_entropy_vec(logits_flat, targets_flat)


def _compute_grads_finite_diff(model, xb, yb, eps=1e-5):
    grads = {}
    params = model.params
    base_loss = _compute_loss(model, xb, yb)
    for name in params.keys():
        p = params[name]
        original = p.copy()
        params[name] = original + eps
        loss_plus = _compute_loss(model, xb, yb)
        params[name] = original - eps
        loss_minus = _compute_loss(model, xb, yb)
        params[name] = original
        grads[name] = ((loss_plus - loss_minus) / (2.0 * eps)).astype(np.float32)
    return grads


def train_epoch(model, lora, opt, tokens, block_size=32, batch_size=2, steps=5):
    initial_weights = {}
    for name in model.params.keys():
        initial_weights[name] = model.params[name].copy()

    total_loss = 0.0
    for step in range(steps):
        xb, yb = _get_batch(tokens, block_size, batch_size)
        loss = _compute_loss(model, xb, yb)
        grads = _compute_grads_finite_diff(model, xb, yb)
        opt.step(grads)
        total_loss += loss

    weight_deltas = {}
    for name in model.params.keys():
        weight_deltas[name] = model.params[name].copy() - initial_weights[name]

    return {
        "weight_deltas": weight_deltas,
        "avg_loss": total_loss / max(1, steps),
    }
