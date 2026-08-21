"""Test Android Python modules standalone (without Android/Chaquopy)."""
import sys
import os

PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "android", "app", "src", "main", "python")
PACKAGE_DIR = os.path.normpath(PACKAGE_DIR)
sys.path.insert(0, PACKAGE_DIR)

from model import NumpyILM, TinyConfig
from train import AdamW, train_epoch
from lora import NumpyLoRA, get_adapter_param_names
from numpy_wire import numpy_to_wire, numpy_from_wire, encode_tensor_dict, decode_tensor_dict
from protocol import generate_nonce, sign_request, verify_request_signature
from ilm_profile import ILMProfile, get_profile, detect_resources, ResourceConfig

import numpy as np

# Test 1: Model forward/backward
print("=== Test 1: Model forward/backward ===")
cfg = TinyConfig(vocab_size=256, block_size=32, n_embd=32, n_head=4, n_layer=2)
model = NumpyILM(cfg)
print(f"Parameters: {model.num_params()}")

x = np.random.randint(0, 255, (2, 16)).astype(np.int64)
y = np.random.randint(0, 255, (2, 16)).astype(np.int64)
logits, loss = model.forward(x, targets=y)
assert np.isfinite(loss), f"Loss is {loss}"
print(f"Forward: loss={loss:.4f}")

model.backward()
for name in model.param_names():
    g = model.grad(name)
    assert g is not None, f"No grad for {name}"
    assert np.all(np.isfinite(g)), f"Bad grad for {name}"
print(f"Backward: {len(model.param_names())} grads OK")

# Test 2: Training loop (returns weight_deltas now)
print("\n=== Test 2: Training loop ===")
opt = AdamW(model.params.data, lr=1e-3)
tokens = np.random.randint(0, 255, 500).astype(np.int64)
result = train_epoch(model, NumpyLoRA(), opt, tokens, block_size=32, batch_size=2, steps=5)
weight_deltas = result["weight_deltas"]
assert len(weight_deltas) > 0, "No weight deltas returned"
print(f"Training: loss={result['avg_loss']:.4f}, deltas={len(weight_deltas)} params")

# Test 3: LoRA + SVD decomposition
print("\n=== Test 3: LoRA SVD decomposition ===")
lora = NumpyLoRA(rank=4, alpha=16)
base_shapes = {n: model.get_param(n).shape for n in model.param_names()}
targets = get_adapter_param_names(model.param_names())
lora.inject(targets, base_shapes)
lora_delta = lora.compute_delta_from_weights(weight_deltas)
assert len(lora_delta) > 0, "No LoRA delta keys"
print(f"LoRA delta: {len(lora_delta)} tensors (keys use server naming)")
for k in sorted(lora_delta.keys())[:3]:
    print(f"  {k}: {lora_delta[k].shape}")

# Test 4: LoRA state dict round-trip (server naming)
print("\n=== Test 4: LoRA state dict round-trip ===")
sd = lora.state_dict()
lora2 = NumpyLoRA(rank=4, alpha=16)
lora2.inject(targets, base_shapes)
lora2.load_state_dict(sd)
for name in lora.lora_a:
    assert np.allclose(lora.lora_a[name], lora2.lora_a[name]), f"LoRA A mismatch: {name}"
print(f"LoRA state dict round-trip OK ({len(sd)} keys)")

# Test 5: Wire format
print("\n=== Test 5: Wire format ===")
arr = np.random.randn(8, 32).astype(np.float32)
wired = numpy_to_wire(arr)
unwired = numpy_from_wire(wired)
assert np.allclose(arr, unwired), "Wire round-trip failed"
print("Wire format OK")

# Test 6: Protocol
print("\n=== Test 6: Protocol ===")
nonce = generate_nonce()
sig = sign_request("secret", "GET", "/v1/round", "client1", nonce, 1000000)
assert verify_request_signature("secret", "GET", "/v1/round", "client1", nonce, 1000000, sig)
print("Protocol HMAC OK")

# Test 7: Model save/load
print("\n=== Test 7: State dict round-trip ===")
model_orig = NumpyILM(TinyConfig(vocab_size=256, block_size=32, n_embd=32, n_head=4, n_layer=2, dropout=0.0))
model_orig.forward(x, targets=y)
sd = model_orig.state_dict()
model_loaded = NumpyILM(TinyConfig(vocab_size=256, block_size=32, n_embd=32, n_head=4, n_layer=2, dropout=0.0))
model_loaded.load_state_dict(sd)
logits_orig, _ = model_orig.forward(x, targets=y)
logits_loaded, _ = model_loaded.forward(x, targets=y)
assert np.allclose(logits_orig, logits_loaded), "State dict round-trip failed"
print("State dict OK")

# Test 8: ILM profiles (torch_threads field)
print("\n=== Test 8: ILM profiles ===")
for name in ("nano", "laptop", "desktop"):
    p = get_profile(name)
    assert hasattr(p, "torch_threads"), f"Profile {name} missing torch_threads"
    print(f"  {name}: embd={p.n_embd}, layers={p.n_layer}, threads={p.torch_threads}")

# Test 9: Resource config auto-detection
print("\n=== Test 9: Resource config ===")
rc = detect_resources()
assert hasattr(rc, "torch_threads"), "ResourceConfig missing torch_threads"
assert rc.block_size >= 256, f"block_size too small: {rc.block_size}"
assert rc.n_embd >= 64, f"n_embd too small: {rc.n_embd}"
assert rc.n_layer >= 2, f"n_layer too small: {rc.n_layer}"
print(f"  auto: block={rc.block_size} embd={rc.n_embd} layers={rc.n_layer} threads={rc.torch_threads}")
print("Resource OK")

print("\n=== ALL TESTS PASSED ===")
