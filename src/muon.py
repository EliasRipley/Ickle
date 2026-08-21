import math
import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim >= 2
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    orig_shape = G.shape
    b, c = G.shape[-2], G.shape[-1]
    if b != c and max(b, c) / min(b, c) > 2:
        denom = G.norm(dim=(-2, -1), keepdim=True) + eps
        result = (G / denom).to(dtype=G.dtype)
        if was_2d:
            result = result.squeeze(0)
        return result
    G = G.reshape(-1, b, c)
    X = G.float().bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if steps > 0 and b == c:
        a_np = X.transpose(-2, -1)
        b_np = torch.eye(b, device=X.device, dtype=X.dtype).expand(len(X), b, b)
        b_np = b_np * (1.0 + 0.0j if X.is_complex() else 1.0)
        for _ in range(steps):
            a_np = 0.5 * (3.0 * a_np - a_np @ b_np @ a_np)
            b_np = 0.5 * (3.0 * b_np - b_np @ a_np @ b_np)
        X = X @ b_np
    result = X.reshape(*orig_shape[:-2], b, c).to(dtype=G.dtype)
    if was_2d:
        result = result.squeeze(0)
    return result


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01,
                 ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1.0 - momentum)
                if nesterov:
                    update = grad.lerp(buf, momentum)
                else:
                    update = buf
                if p.ndim >= 2 and update.size(-2) <= 8192 and update.size(-1) <= 8192:
                    if update.size(-2) >= 8 and update.size(-1) >= 8:
                        update = zeropower_via_newtonschulz5(update, steps=ns)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class MuonWithAuxAdam:
    def __init__(self, param_groups):
        self.muon_groups = []
        self.adam_groups = []
        self._muon = None
        self._adam = None
        for pg in param_groups:
            pg = dict(pg)
            if pg.pop("use_muon", False):
                self.muon_groups.append(pg)
            else:
                self.adam_groups.append(pg)
        self._built = False

    def _build_children(self):
        if self._built:
            return
        self._built = True
        muon_params = []
        for pg in self.muon_groups:
            muon_dict = {
                "params": pg["params"],
                "lr": pg.get("lr", 0.02),
                "momentum": pg.get("momentum", 0.95),
                "weight_decay": pg.get("weight_decay", 0.01),
                "ns_steps": pg.get("ns_steps", 5),
                "nesterov": pg.get("nesterov", True),
            }
            muon_params.append(muon_dict)
        adam_params = []
        for pg in self.adam_groups:
            adam_dict = {
                "params": pg["params"],
                "lr": pg.get("lr", 3e-4),
                "betas": pg.get("betas", (0.9, 0.95)),
                "eps": pg.get("eps", 1e-8),
                "weight_decay": pg.get("weight_decay", 0.01),
            }
            adam_params.append(adam_dict)
        self._muon = Muon(muon_params) if muon_params else None
        self._adam = torch.optim.AdamW(adam_params, fused=False) if adam_params else None

    @torch.no_grad()
    def step(self, closure=None):
        self._build_children()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon:
            self._muon.step()
        if self._adam:
            self._adam.step()
        return loss

    def state_dict(self):
        self._build_children()
        return {
            "muon": self._muon.state_dict() if self._muon else None,
            "adam": self._adam.state_dict() if self._adam else None,
        }

    def load_state_dict(self, sd):
        self._build_children()
        if sd.get("muon") and self._muon:
            self._muon.load_state_dict(sd["muon"])
        if sd.get("adam") and self._adam:
            self._adam.load_state_dict(sd["adam"])

    @property
    def param_groups(self):
        self._build_children()
        groups = []
        if self._muon:
            groups.extend(self._muon.param_groups)
        if self._adam:
            groups.extend(self._adam.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        if self._muon:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adam:
            self._adam.zero_grad(set_to_none=set_to_none)
