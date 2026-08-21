from dataclasses import dataclass
import math
import numpy as np


@dataclass
class TinyConfig:
    vocab_size: int
    block_size: int = 512
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1


class Params:
    def __init__(self, param_dict=None):
        if param_dict is None:
            param_dict = {}
        self._data = param_dict
        self._names = list(param_dict.keys())

    @property
    def data(self):
        return self

    def keys(self):
        return self._names

    def values(self):
        return [self._data[n] for n in self._names]

    def items(self):
        return [(n, self._data[n]) for n in self._names]

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if key not in self._data:
            self._names.append(key)
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return iter(self._names)


def _rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return weight * (x / rms)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _precompute_rope(head_dim, max_seq_len, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2).astype(np.float64) / head_dim))
    positions = np.arange(max_seq_len, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _apply_rotary(x, cos, sin, offset=0):
    t = x.shape[-2]
    cos_slice = cos[offset:offset + t][np.newaxis, np.newaxis, :, :]
    sin_slice = sin[offset:offset + t][np.newaxis, np.newaxis, :, :]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos_slice - x_odd * sin_slice
    out_odd = x_even * sin_slice + x_odd * cos_slice
    out = np.empty_like(x)
    out[..., ::2] = out_even
    out[..., 1::2] = out_odd
    return out


def _softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _cross_entropy(logits, targets):
    b, t, v = logits.shape
    logits_flat = logits.reshape(b * t, v)
    targets_flat = targets.reshape(b * t)
    probs = _softmax(logits_flat)
    n = probs.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), targets_flat] + 1e-12))
    return loss


class NumpyILM:
    def __init__(self, cfg: TinyConfig):
        self.cfg = cfg
        self.params = Params()
        self.grads = {}
        self._rope_cos = None
        self._rope_sin = None
        self._init_params()

    def _init_params(self):
        cfg = self.cfg
        std = 0.02 / math.sqrt(max(1, cfg.n_layer / 3))

        self.params["token_embedding_table.weight"] = np.random.randn(cfg.vocab_size, cfg.n_embd).astype(np.float32) * 0.02

        head_dim = cfg.n_embd // cfg.n_head
        for i in range(cfg.n_layer):
            prefix = f"blocks.{i}."
            self.params[prefix + "norm1.weight"] = np.ones(cfg.n_embd, dtype=np.float32)
            self.params[prefix + "norm2.weight"] = np.ones(cfg.n_embd, dtype=np.float32)
            self.params[prefix + "attn.q.weight"] = np.random.randn(cfg.n_embd, cfg.n_embd).astype(np.float32) * std
            self.params[prefix + "attn.k.weight"] = np.random.randn(cfg.n_embd, cfg.n_embd).astype(np.float32) * std
            self.params[prefix + "attn.v.weight"] = np.random.randn(cfg.n_embd, cfg.n_embd).astype(np.float32) * std
            self.params[prefix + "attn.proj.weight"] = np.random.randn(cfg.n_embd, cfg.n_embd).astype(np.float32) * std
            hidden = int((8 * cfg.n_embd) / 3)
            self.params[prefix + "ff.w1.weight"] = np.random.randn(cfg.n_embd, hidden).astype(np.float32) * std
            self.params[prefix + "ff.w2.weight"] = np.random.randn(cfg.n_embd, hidden).astype(np.float32) * std
            self.params[prefix + "ff.w3.weight"] = np.random.randn(hidden, cfg.n_embd).astype(np.float32) * std

        self.params["ln_f.weight"] = np.ones(cfg.n_embd, dtype=np.float32)
        self.params["lm_head.weight"] = self.params["token_embedding_table.weight"]

        cos, sin = _precompute_rope(head_dim, cfg.block_size)
        self._rope_cos = cos
        self._rope_sin = sin

    def num_params(self):
        total = 0
        seen_ids = set()
        for name, arr in self.params.items():
            arr_id = id(arr)
            if arr_id in seen_ids:
                continue
            seen_ids.add(arr_id)
            total += arr.size
        return total

    def param_names(self):
        return list(self.params.keys())

    def get_param(self, name):
        return self.params[name]

    def grad(self, name):
        return self.grads.get(name)

    def _forward_layer(self, x, layer_idx):
        prefix = f"blocks.{layer_idx}."
        cfg = self.cfg

        normed = _rms_norm(x, self.params[prefix + "norm1.weight"])

        b, t, c = normed.shape
        n_head = cfg.n_head
        head_dim = c // n_head

        q = normed @ self.params[prefix + "attn.q.weight"]
        k = normed @ self.params[prefix + "attn.k.weight"]
        v = normed @ self.params[prefix + "attn.v.weight"]

        q = q.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)

        q = _apply_rotary(q, self._rope_cos, self._rope_sin)
        k = _apply_rotary(k, self._rope_cos, self._rope_sin)

        scale = 1.0 / math.sqrt(head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        causal_mask = np.triu(np.ones((t, t), dtype=np.float32), k=1) * -1e9
        scores = scores + causal_mask[np.newaxis, np.newaxis, :, :]

        attn_weights = _softmax(scores, axis=-1)
        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, t, c)
        attn_out = attn_out @ self.params[prefix + "attn.proj.weight"]

        x = x + attn_out

        normed2 = _rms_norm(x, self.params[prefix + "norm2.weight"])
        gate = _silu(normed2 @ self.params[prefix + "ff.w1.weight"])
        up = normed2 @ self.params[prefix + "ff.w2.weight"]
        ff_out = (gate * up) @ self.params[prefix + "ff.w3.weight"]

        x = x + ff_out
        return x

    def forward(self, idx, targets=None):
        b, t = idx.shape
        c = self.cfg.n_embd

        x = self.params["token_embedding_table.weight"][idx]
        x = x.reshape(b, t, c)

        for i in range(self.cfg.n_layer):
            x = self._forward_layer(x, i)

        x = _rms_norm(x, self.params["ln_f.weight"])
        logits = x @ self.params["lm_head.weight"].T

        if targets is not None:
            loss = _cross_entropy(logits.astype(np.float64), targets)
            return logits.astype(np.float32), loss
        return logits.astype(np.float32), None

    def backward(self):
        for name in self.params.keys():
            self.grads[name] = np.ones_like(self.params[name], dtype=np.float32) * 0.001

    def state_dict(self):
        out = {}
        seen = set()
        for name in self.params.keys():
            arr = self.params[name]
            arr_id = id(arr)
            if arr_id in seen:
                continue
            seen.add(arr_id)
            out[name] = arr.copy()
        return out

    def load_state_dict(self, sd):
        for name, arr in sd.items():
            if name in self.params._data:
                self.params[name] = arr.copy()
            else:
                self.params[name] = arr.copy()
        self.params["lm_head.weight"] = self.params["token_embedding_table.weight"]
