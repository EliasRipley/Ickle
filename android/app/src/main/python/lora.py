import numpy as np


TARGET_MODULES = {"q", "k", "v", "proj", "w1", "w2", "w3", "lm_head"}


class NumpyLoRA:
    def __init__(self, rank=4, alpha=16):
        self.rank = rank
        self.alpha = alpha
        self.scale = float(alpha) / max(1, int(rank))
        self.targets = []
        self.base_shapes = {}
        self.lora_a = {}
        self.lora_b = {}

    def inject(self, targets, base_shapes):
        self.targets = list(targets)
        self.base_shapes = dict(base_shapes)
        self.lora_a = {}
        self.lora_b = {}
        for name in self.targets:
            shape = self.base_shapes.get(name)
            if shape is None or len(shape) != 2:
                continue
            out_features, in_features = shape
            r = min(self.rank, out_features, in_features)
            lora_a = np.random.randn(r, in_features).astype(np.float32) * 0.02
            lora_b = np.zeros((out_features, r), dtype=np.float32)
            self.lora_a[name] = lora_a
            self.lora_b[name] = lora_b

    def compute_delta_from_weights(self, weight_deltas):
        out = {}
        for name in self.targets:
            if name not in weight_deltas:
                continue
            delta = np.array(weight_deltas[name], dtype=np.float32)
            out_features, in_features = delta.shape
            r = min(self.rank, out_features, in_features)
            try:
                u, s, vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
                u = u[:, :r].astype(np.float32)
                s_diag = np.diag(np.sqrt(s[:r].astype(np.float32)))
                vt = vt[:r, :].astype(np.float32)
                lora_a = (s_diag @ vt).astype(np.float32) * self.scale
                lora_b = (u @ s_diag).astype(np.float32)
            except np.linalg.LinAlgError:
                lora_a = np.zeros((r, in_features), dtype=np.float32)
                lora_b = np.zeros((out_features, r), dtype=np.float32)
            out[f"{name}.lora_a"] = lora_a
            out[f"{name}.lora_b"] = lora_b
        return out

    def state_dict(self):
        sd = {}
        for name, arr in self.lora_a.items():
            sd[f"{name}.lora_a"] = arr.copy()
        for name, arr in self.lora_b.items():
            sd[f"{name}.lora_b"] = arr.copy()
        return sd

    def load_state_dict(self, sd):
        self.lora_a = {}
        self.lora_b = {}
        for key, arr in sd.items():
            if key.endswith(".lora_a"):
                name = key[:-len(".lora_a")]
                self.lora_a[name] = arr.copy()
            elif key.endswith(".lora_b"):
                name = key[:-len(".lora_b")]
                self.lora_b[name] = arr.copy()


def get_adapter_param_names(param_names):
    out = []
    for name in param_names:
        shortened = name.replace(".weight", "")
        parts = shortened.rsplit(".", 1)
        module = parts[-1] if len(parts) > 1 else shortened
        if module in TARGET_MODULES:
            out.append(name)
    return out
