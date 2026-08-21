from __future__ import annotations

import argparse
import hmac
import json
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.federated.coordinator import AggregationOptions, FederatedCoordinator
from src.federated.lora import LoRAConfig
from src.federated.protocol import generate_nonce, now_epoch_seconds


class FederatedRequestHandler(BaseHTTPRequestHandler):
    server_version = "IckleFederated/0.1"

    @property
    def coordinator(self) -> FederatedCoordinator:
        return self.server.coordinator  # type: ignore[attr-defined]

    @property
    def admin_secret(self) -> str:
        return self.server.admin_secret  # type: ignore[attr-defined]

    @property
    def registration_secret(self) -> str:
        return self.server.registration_secret  # type: ignore[attr-defined]

    @property
    def max_request_bytes(self) -> int:
        return self.server.max_request_bytes  # type: ignore[attr-defined]

    def _json_response(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type,Authorization,X-Admin-Secret,X-Registration-Secret",
        )
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0:
            raise ValueError("Invalid Content-Length")
        if length > self.max_request_bytes:
            raise OverflowError(
                f"Request body exceeds {self.max_request_bytes} byte limit"
            )
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type,Authorization,X-Admin-Secret,X-Registration-Secret",
        )
        self.end_headers()

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json_response(200, {"ok": True})
            return
        if parsed.path == "/v1/status":
            self._json_response(200, self.coordinator.status())
            return
        if parsed.path == "/v1/round":
            qs = parse_qs(parsed.query)
            client_id = qs.get("client_id", [""])[0]
            nonce = qs.get("nonce", [""])[0]
            timestamp_str = qs.get("timestamp", [""])[0]
            signature = qs.get("signature", [""])[0]
            if not client_id or not nonce or not timestamp_str or not signature:
                self._json_response(400, {"error": "Missing required query params: client_id, nonce, timestamp, signature"})
                return
            try:
                timestamp = int(timestamp_str)
                self.coordinator.verify_round_request(
                    client_id=client_id,
                    nonce=nonce,
                    timestamp=timestamp,
                    signature=signature,
                )
                payload = self.coordinator.get_round_payload(client_id)
                self._json_response(200, payload)
            except Exception as exc:  # noqa: BLE001
                self._json_response(400, {"error": str(exc)})
            return
        self._json_response(404, {"error": "Not found"})

    def _check_admin(self) -> bool:
        if not self.admin_secret:
            return True
        return hmac.compare_digest(self.headers.get("X-Admin-Secret", ""), self.admin_secret)

    def _check_registration(self) -> bool:
        if not self.registration_secret:
            return True
        return hmac.compare_digest(
            self.headers.get("X-Registration-Secret", ""), self.registration_secret
        )

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except OverflowError as exc:
            self._json_response(413, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            self._json_response(400, {"error": f"Invalid JSON: {exc}"})
            return

        if parsed.path == "/v1/register":
            if not self._check_registration():
                self._json_response(403, {"error": "Forbidden: invalid or missing registration secret"})
                return
            platform = str(payload.get("platform", "unknown"))
            device_name = str(payload.get("device_name", ""))
            capabilities = payload.get("capabilities", {})
            try:
                result = self.coordinator.register_client(
                    platform=platform,
                    device_name=device_name,
                    capabilities=capabilities if isinstance(capabilities, dict) else {},
                )
                self._json_response(200, result)
            except PermissionError as exc:
                self._json_response(403, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._json_response(400, {"error": str(exc)})
            return

        if parsed.path == "/v1/submit":
            try:
                result = self.coordinator.submit_update(payload)
                if result.get("ready_for_aggregation"):
                    agg = self.coordinator.aggregate_active_round(force=False)
                    result["aggregation"] = agg
                self._json_response(200, result)
            except PermissionError as exc:
                self._json_response(403, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._json_response(400, {"error": str(exc)})
            return

        if parsed.path == "/v1/aggregate":
            if not self._check_admin():
                self._json_response(403, {"error": "Forbidden: invalid or missing admin secret"})
                return
            force = bool(payload.get("force", False))
            try:
                summary = self.coordinator.aggregate_active_round(force=force)
                self._json_response(200, summary)
            except Exception as exc:  # noqa: BLE001
                self._json_response(400, {"error": str(exc)})
            return

        self._json_response(404, {"error": "Not found"})


def _parse_target_modules(csv_text: str) -> tuple[str, ...]:
    parts = [chunk.strip() for chunk in csv_text.split(",")]
    return tuple([p for p in parts if p])


def main():
    parser = argparse.ArgumentParser(description="Ickle federated coordinator + mobile bridge server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--base-model", default=os.environ.get("ICKLE_MODEL_PATH", ""))
    parser.add_argument("--state-dir", default="data/federated")
    parser.add_argument("--min-clients", type=int, default=3)
    parser.add_argument("--aggregation", default="trimmed_mean", choices=["trimmed_mean", "weighted_avg", "krum"])
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--max-update-norm", type=float, default=5.0)
    parser.add_argument("--byzantine-f", type=int, default=1)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--eval-iters", type=int, default=8)
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="q,k,v,proj,w1,w2,w3,lm_head")
    parser.add_argument("--cert", default="", help="Path to TLS certificate file (enables HTTPS)")
    parser.add_argument("--key", default="", help="Path to TLS private key file")
    parser.add_argument("--diloco-local-steps", type=int, default=100, help="DiLoCo: local training steps per round on each client")
    parser.add_argument("--diloco-lr", type=float, default=1.0, help="DiLoCo: global optimizer learning rate")
    parser.add_argument("--diloco-beta", type=float, default=0.9, help="DiLoCo: global optimizer momentum factor")
    parser.add_argument("--diloco-nesterov", action="store_true", default=True, help="DiLoCo: use Nesterov momentum")
    parser.add_argument("--admin-secret", default="", help="Secret required for POST /v1/aggregate (empty = no auth)")
    parser.add_argument(
        "--registration-secret",
        default=os.environ.get("ICKLE_REGISTRATION_SECRET", ""),
        help="Optional shared secret required to register a contributor",
    )
    parser.add_argument(
        "--max-request-mib",
        type=int,
        default=16,
        help="Maximum JSON request size in MiB (default 16)",
    )
    parser.add_argument(
        "--max-examples-per-update",
        type=int,
        default=100_000,
        help="Reject a client-reported sample count above this value",
    )
    parser.add_argument(
        "--allow-insecure-public",
        action="store_true",
        help="Explicitly allow a non-loopback HTTP listener without TLS (development only)",
    )
    args = parser.parse_args()
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts and not (args.cert and args.key) and not args.allow_insecure_public:
        raise SystemExit(
            "Refusing to expose contributor secrets over public plaintext HTTP. "
            "Configure --cert and --key, bind to 127.0.0.1 behind a TLS proxy, "
            "or explicitly use --allow-insecure-public for isolated development."
        )
    if not args.base_model:
        from src.model_resolver import resolve_default_model

        try:
            args.base_model = resolve_default_model()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None

    lora_cfg = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=_parse_target_modules(args.lora_targets),
    )
    agg_cfg = AggregationOptions(
        method=args.aggregation,
        trim_ratio=args.trim_ratio,
        max_update_norm=args.max_update_norm,
        byzantine_f=args.byzantine_f,
    )
    coordinator = FederatedCoordinator(
        base_model_path=args.base_model,
        state_dir=args.state_dir,
        lora_cfg=lora_cfg,
        min_clients=args.min_clients,
        aggregation=agg_cfg,
        eval_data_path=args.eval_data,
        eval_iters=args.eval_iters,
        max_regression=args.max_regression,
        diloco_local_steps=args.diloco_local_steps,
        diloco_lr=args.diloco_lr,
        diloco_beta=args.diloco_beta,
        diloco_nesterov=args.diloco_nesterov,
        max_examples_per_update=args.max_examples_per_update,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), FederatedRequestHandler)
    httpd.coordinator = coordinator  # type: ignore[attr-defined]
    httpd.admin_secret = args.admin_secret  # type: ignore[attr-defined]
    httpd.registration_secret = args.registration_secret  # type: ignore[attr-defined]
    httpd.max_request_bytes = max(1, int(args.max_request_mib)) * 1024 * 1024  # type: ignore[attr-defined]

    use_tls = bool(args.cert and args.key)
    if use_tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
        print(f"TLS enabled (cert={args.cert})")
    else:
        scheme = "http"
        print("WARNING: TLS not enabled. Communication is in plaintext.")

    print(f"Ickle federated server listening on {scheme}://{args.host}:{args.port}")
    print("Endpoints: /health, /v1/status, /v1/register, /v1/round, /v1/submit, /v1/aggregate")
    if args.admin_secret:
        print("Admin secret configured for /v1/aggregate (send X-Admin-Secret header)")
    if args.registration_secret:
        print("Registration admission control is enabled (send X-Registration-Secret header)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down federated server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
