import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from protocol import canonical_json, generate_nonce, sign_payload, sign_request
from numpy_wire import encode_tensor_dict, decode_tensor_dict

_now_epoch = __import__("time").time


def now_epoch_seconds():
    return int(_now_epoch())


def _http_json(method, url, payload=None):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        exc.close()
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc


def register(server_url, platform="android", device_name="", data_dir="/data/local/tmp/ickle"):
    result = _http_json(
        "POST",
        f"{server_url}/v1/register",
        {
            "platform": platform,
            "device_name": device_name,
            "capabilities": {
                "local_adapter_training": True,
            },
        },
    )
    identity = {
        "client_id": result["client_id"],
        "client_secret": result["client_secret"],
        "platform": platform,
        "device_name": device_name,
    }
    os.makedirs(data_dir, exist_ok=True)
    identity_path = os.path.join(data_dir, "client_identity.json")
    with open(identity_path, "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2)
    return result


def fetch_round(server_url, client_id, client_secret):
    nonce = generate_nonce()
    timestamp = now_epoch_seconds()
    signature = sign_request(client_secret, "GET", "/v1/round", client_id, nonce, timestamp)
    qs = urllib.parse.urlencode({
        "client_id": client_id,
        "nonce": nonce,
        "timestamp": str(timestamp),
        "signature": signature,
    })
    return _http_json("GET", f"{server_url}/v1/round?{qs}")


def submit(server_url, client_id, client_secret, round_id, metrics, delta):
    envelope = {
        "client_id": client_id,
        "round_id": round_id,
        "num_examples": int(metrics.get("token_count", 1)),
        "metrics": metrics,
        "delta": encode_tensor_dict(delta),
        "timestamp": now_epoch_seconds(),
        "nonce": generate_nonce(),
    }
    envelope["signature"] = sign_payload(client_secret, envelope)
    return _http_json("POST", f"{server_url}/v1/submit", envelope)


def load_identity(data_dir="/data/local/tmp/ickle"):
    identity_path = os.path.join(data_dir, "client_identity.json")
    if not os.path.exists(identity_path):
        return {}
    with open(identity_path, "r", encoding="utf-8") as f:
        return json.load(f)
