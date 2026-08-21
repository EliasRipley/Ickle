import base64
import numpy as np


def numpy_to_wire(arr):
    data = base64.b64encode(arr.astype(np.float32).tobytes(order="C")).decode("ascii")
    return {
        "dtype": "float32",
        "shape": list(arr.shape),
        "data_b64": data,
    }


def numpy_from_wire(payload):
    dtype = payload.get("dtype", "float32")
    if dtype != "float32":
        raise ValueError(f"Unsupported tensor dtype '{dtype}'")
    shape = payload.get("shape", [])
    raw = base64.b64decode(payload["data_b64"].encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def encode_tensor_dict(tensors):
    return {name: numpy_to_wire(arr) for name, arr in tensors.items()}


def decode_tensor_dict(payload):
    out = {}
    for name, wire in payload.items():
        out[name] = numpy_from_wire(wire)
    return out
