import json
import os
import sys

from model import NumpyILM, TinyConfig
from train import AdamW, train_epoch
from lora import NumpyLoRA, get_adapter_param_names
from numpy_wire import encode_tensor_dict
from protocol import sign_payload, generate_nonce
from ilm_profile import get_profile
from client import register as client_register, fetch_round, submit, load_identity, now_epoch_seconds
from torickle_numpy import pack_delta
from swarm_identity import ensure_identity, load_identity as load_swarm_identity
from swarm_announce import BundleAnnouncement, announce_to_swarm


class IckleMobileClient:
    def __init__(self, server_url="http://127.0.0.1:8788", data_dir="/data/local/tmp/ickle",
                 device_name="android", profile_name="nano", seed_text="The quick brown fox",
                 auto_torickle=False, swarm_host="", swarm_port=8790):
        self.server_url = server_url.rstrip("/")
        self.data_dir = data_dir
        self.device_name = device_name
        self.profile = get_profile(profile_name)
        self.seed_text = seed_text
        self.auto_torickle = auto_torickle
        self.swarm_host = swarm_host
        self.swarm_port = int(swarm_port)
        self._swarm_identity = None
        os.makedirs(data_dir, exist_ok=True)

        identity = load_identity(data_dir)
        self.client_id = identity.get("client_id", "")
        self.client_secret = identity.get("client_secret", "")
        self._model_cache = None

    def is_registered(self):
        return bool(self.client_id and self.client_secret)

    def register(self):
        result = client_register(
            self.server_url,
            platform="android",
            device_name=self.device_name,
            data_dir=self.data_dir,
        )
        identity = load_identity(self.data_dir)
        self.client_id = identity["client_id"]
        self.client_secret = identity["client_secret"]
        return result

    def do_round(self):
        if not self.is_registered():
            return {"error": "not registered"}

        round_data = fetch_round(self.server_url, self.client_id, self.client_secret)
        round_id = round_data["round_id"]
        lora_cfg = round_data.get("lora_config", {})
        rank = lora_cfg.get("rank", 4)
        alpha = lora_cfg.get("alpha", 16)
        global_adapter = round_data.get("global_adapter", {})

        result = self._train_local(round_data, rank, alpha)
        metrics = result["metrics"]
        lora_delta = result["lora_delta"]

        submit_result = submit(
            self.server_url,
            self.client_id,
            self.client_secret,
            round_id,
            metrics,
            lora_delta,
        )

        if self.auto_torickle and lora_delta:
            submit_result["torickle"] = self._pack_and_announce(
                lora_delta, round_id, submit_result
            )

        # The server's submit acknowledgment doesn't echo back this round's
        # local training metrics -- merge them in so callers (the Android
        # foreground service and, through it, the in-app stats display) see
        # what actually happened on-device, not just "accepted".
        submit_result.setdefault("round_id", round_id)
        submit_result["metrics"] = metrics
        return submit_result

    def _pack_and_announce(self, lora_delta, round_id, submit_result):
        torickle_info = {}
        try:
            bundle_dir = os.path.join(self.data_dir, "torickle", f"round_{round_id}")
            pack_result = pack_delta(
                delta=lora_delta,
                out_dir=bundle_dir,
                overwrite=True,
                metadata={
                    "round_id": str(round_id),
                    "client_id": self.client_id,
                    "platform": "android",
                    "device_name": self.device_name,
                },
            )
            torickle_info = {
                "packed": True,
                "manifest_path": pack_result["manifest_path"],
                "piece_count": pack_result["piece_count"],
                "total_bytes": pack_result["total_bytes"],
                "payload_sha256": pack_result["payload_sha256"],
                "merkle_root": pack_result["merkle_root"],
            }
        except Exception as e:
            return {"packed": False, "error": str(e)}

        if self.swarm_host:
            torickle_info["swarm_announced"] = False
            try:
                if self._swarm_identity is None:
                    identity_path = os.path.join(self.data_dir, "torickle", "swarm_identity.json")
                    self._swarm_identity = ensure_identity(identity_path, label=self.device_name)

                ann = BundleAnnouncement(
                    bundle_id=pack_result["payload_sha256"][:16],
                    model_hash="",
                    piece_count=pack_result["piece_count"],
                    total_bytes=pack_result["total_bytes"],
                    payload_sha256=pack_result["payload_sha256"],
                    merkle_root=pack_result["merkle_root"],
                    peer_id=self._swarm_identity.peer_id,
                    host=self.swarm_host,
                    port=self.swarm_port,
                )
                ann.sign(self._swarm_identity)

                response = announce_to_swarm(ann, self.swarm_host, self.swarm_port)
                torickle_info["swarm_announced"] = response.get("accepted", False)
                torickle_info["swarm_response"] = response
            except Exception as e:
                torickle_info["swarm_error"] = str(e)

        return torickle_info

    def _train_local(self, round_data, rank, alpha):
        p = self.profile
        cfg = TinyConfig(
            vocab_size=256,
            block_size=p.block_size,
            n_embd=p.n_embd,
            n_head=p.n_head,
            n_layer=p.n_layer,
            dropout=0.0,
        )
        model = NumpyILM(cfg)
        self._model_cache = model

        text = self._load_training_text()
        tokens = [min(ord(c) % 256, 255) for c in text]
        if len(tokens) < p.block_size + 1:
            tokens = (tokens * ((p.block_size + 1) // len(tokens) + 1))[:p.block_size + 1]

        tokens_arr = __import__("numpy").array(tokens, dtype=__import__("numpy").int64)

        lora = NumpyLoRA(rank=rank, alpha=alpha)
        base_shapes = {n: model.get_param(n).shape for n in model.param_names()}
        targets = get_adapter_param_names(model.param_names())
        lora.inject(targets, base_shapes)

        opt = AdamW(model.params, lr=1e-3)

        train_result = train_epoch(
            model, lora, opt, tokens_arr,
            block_size=min(p.block_size // 2, len(tokens) - 1),
            batch_size=min(p.batch_size, 4),
            steps=10,
        )

        lora_delta = lora.compute_delta_from_weights(train_result["weight_deltas"])

        metrics = {
            "final_loss": float(train_result["avg_loss"]),
            "token_count": len(tokens),
        }
        return {"metrics": metrics, "lora_delta": lora_delta}

    def _load_training_text(self):
        corpus_path = os.path.join(self.data_dir, "training_corpus.txt")
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
            if len(text.strip()) > 100:
                return text
        return self.seed_text * 100


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ickle Mobile Federated Client")
    parser.add_argument("command", choices=["register", "round"])
    parser.add_argument("--server", default="http://127.0.0.1:8788")
    parser.add_argument("--data-dir", default="/data/local/tmp/ickle")
    parser.add_argument("--device-name", default="android")
    parser.add_argument("--profile", default="nano", choices=["nano", "laptop", "desktop"])
    parser.add_argument("--auto-torickle", action="store_true", help="Pack trained deltas into torickle bundles")
    parser.add_argument("--swarm-host", default="", help="Swarm node host for announcing torickle bundles")
    parser.add_argument("--swarm-port", type=int, default=8790, help="Swarm node port (default 8790)")
    args = parser.parse_args()

    c = IckleMobileClient(
        server_url=args.server,
        data_dir=args.data_dir,
        device_name=args.device_name,
        profile_name=args.profile,
        auto_torickle=bool(args.auto_torickle),
        swarm_host=args.swarm_host,
        swarm_port=args.swarm_port,
    )

    if args.command == "register":
        result = c.register()
        print(json.dumps(result, indent=2))
    elif args.command == "round":
        result = c.do_round()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
