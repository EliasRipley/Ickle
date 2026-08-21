from __future__ import annotations

import argparse
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import uuid

import torch

from src.federated.contribution_ledger import DEFAULT_LEDGER_PATH, LedgerStore
from src.federated.identity import ensure_identity
from src.federated.local_train import (
    LocalTrainConfig,
    StreamDataConfig,
    evaluate_adapter_loss,
    load_training_text,
    train_local_delta,
)
from src.federated.lora import LoRAConfig, add_states
from src.federated.protocol import (
    decode_tensor_dict,
    encode_tensor_dict,
    file_sha256,
    generate_nonce,
    now_epoch_seconds,
    sign_payload,
    sign_request,
)
from src.federated.swarm import SwarmNode, DEFAULT_DATA_DIR, DEFAULT_IDENTITY_PATH
from src.torickle import pack_delta_file
from src.workspace_paths import get_training_root


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    verify_tls: bool = True,
) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    parsed = urllib.parse.urlparse(url)
    ctx = None
    if parsed.scheme == "https":
        if verify_tls:
            ctx = ssl.create_default_context()
        else:
            ctx = ssl._create_unverified_context()  # noqa: SLF001
    try:
        if ctx is None:
            resp_ctx = urllib.request.urlopen(req, timeout=120)
        else:
            resp_ctx = urllib.request.urlopen(req, timeout=120, context=ctx)
        with resp_ctx as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        exc.close()
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc


def _load_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_identity(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _register(server_url: str, platform: str, device_name: str, *, verify_tls: bool) -> dict[str, Any]:
    return _http_json(
        "POST",
        f"{server_url}/v1/register",
        {
            "platform": platform,
            "device_name": device_name,
            "capabilities": {
                "python_client": True,
                "local_adapter_training": True,
            },
        },
        verify_tls=verify_tls,
    )


def _fetch_round(server_url: str, client_id: str, client_secret: str, *, verify_tls: bool) -> dict[str, Any]:
    nonce = generate_nonce()
    timestamp = now_epoch_seconds()
    signature = sign_request(client_secret, "GET", "/v1/round", client_id, nonce, timestamp)
    qs = urllib.parse.urlencode({
        "client_id": client_id,
        "nonce": nonce,
        "timestamp": str(timestamp),
        "signature": signature,
    })
    return _http_json("GET", f"{server_url}/v1/round?{qs}", verify_tls=verify_tls)


def _pick_training_corpus(training_root: Path) -> Path | None:
    preferred = [
        "combined_corpus.txt",
        "base_lm_corpus_v2.txt",
        "ickle_v3_corpus.txt",
        "open_fineweb_edu_stream.txt",
    ]
    for name in preferred:
        candidate = training_root / name
        if candidate.exists() and candidate.is_file():
            return candidate
    txt_files = sorted([p for p in training_root.glob("*.txt") if p.is_file()])
    if txt_files:
        return txt_files[0]
    return None


def _resolve_local_data_path(
    explicit_local_data: str,
    *,
    training_root: Path,
    prefer_training_root_data: bool,
) -> tuple[Path, str]:
    if explicit_local_data.strip():
        return Path(explicit_local_data).resolve(), "explicit"

    if prefer_training_root_data:
        picked = _pick_training_corpus(training_root)
        if picked is not None:
            return picked.resolve(), "training_root"

    legacy_candidates = [
        (training_root / "corpuses" / "ickle_clean_corpus.txt").resolve(),
        (Path("data") / "ickle_clean_corpus.txt").resolve(),
    ]
    for legacy_default in legacy_candidates:
        if legacy_default.exists() and legacy_default.is_file():
            return legacy_default, "legacy_default"

    picked = _pick_training_corpus(training_root)
    if picked is not None:
        return picked.resolve(), "training_root_fallback"
    return legacy_candidates[0], "legacy_default_missing"


def _resolve_eval_data_path(explicit_eval_data: str, *, local_data_path: Path) -> Path:
    if explicit_eval_data.strip():
        return Path(explicit_eval_data).resolve()
    return local_data_path


def _stream_descriptor(cfg: StreamDataConfig) -> str:
    config_part = f":{cfg.config}" if cfg.config else ""
    return f"hf://{cfg.dataset}{config_part}/{cfg.split}"


def main():
    parser = argparse.ArgumentParser(description="Ickle federated desktop client")
    parser.add_argument("--server", default="http://127.0.0.1:8788")
    parser.add_argument("--identity", default="data/federated/client_identity.json")
    parser.add_argument("--platform", default="desktop")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--local-data", default="")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument(
        "--prefer-training-root-data",
        action="store_true",
        help="When --local-data is omitted, prefer corpora from IckleTraining over runtime data/.",
    )
    parser.add_argument(
        "--source-mode",
        default="auto",
        choices=["auto", "stream", "local"],
        help="Training data source: stream-first auto mode, stream-only, or local-only.",
    )
    parser.add_argument("--stream-dataset", default="HuggingFaceFW/fineweb-edu", help="HF dataset id for stream-first training.")
    parser.add_argument("--stream-config", default="", help="Optional HF dataset config/subset name.")
    parser.add_argument("--stream-split", default="train", help="HF split for streaming.")
    parser.add_argument("--stream-field", default="text", help="Field to read from streamed dataset rows.")
    parser.add_argument("--stream-filter", default="", help="Optional row filter expression, e.g. row.get('language')=='en'.")
    parser.add_argument("--stream-max-chars", type=int, default=1_200_000, help="Maximum streamed chars to consume per round.")
    parser.add_argument("--stream-min-chars", type=int, default=1200, help="Minimum streamed chars required before training.")
    parser.add_argument("--base-model", default=os.environ.get("ICKLE_MODEL_PATH", ""))
    parser.add_argument("--steps", type=int, default=None, help="Local training steps (defaults to server's diloco_local_steps)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--burst-seconds", type=float, default=0.0, help="Optional wall-clock budget for burst training.")
    parser.add_argument("--min-steps", type=int, default=1, help="Minimum local steps before burst timeout can stop training.")
    parser.add_argument(
        "--require-local-improvement",
        action="store_true",
        help="Require local eval loss improvement before submitting update.",
    )
    parser.add_argument(
        "--min-loss-improvement",
        type=float,
        default=0.0,
        help="Minimum required local eval loss reduction (before - after) to submit.",
    )
    parser.add_argument("--local-eval-data", default="", help="Optional eval corpus path (defaults to --local-data).")
    parser.add_argument("--local-eval-iters", type=int, default=3)
    parser.add_argument("--auto-torickle", action="store_true", help="Auto-pack the trained delta into a torickle bundle and announce via swarm.")
    parser.add_argument("--model-hash", default="", help="Model hash for torickle announcement (used with --auto-torickle).")
    parser.add_argument(
        "--allow-base-model-mismatch",
        action="store_true",
        help="Allow running even if server base_model_sha256 does not match local --base-model file.",
    )
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Disable TLS certificate verification for HTTPS (unsafe; development only).",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.base_model:
        from src.model_resolver import resolve_default_model

        try:
            args.base_model = resolve_default_model()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None

    server_url = args.server.rstrip("/")
    verify_tls = not bool(args.insecure_skip_tls_verify)
    identity_path = Path(args.identity)
    training_root = Path(args.training_root).resolve()
    prefer_training_root_data = bool(args.prefer_training_root_data)
    source_mode = str(args.source_mode or "auto").strip().lower()
    explicit_local_data = str(args.local_data or "").strip()

    local_data_path: Path | None = None
    stream_cfg: StreamDataConfig | None = None
    local_data_origin = ""
    local_data_display = ""
    training_text = ""
    training_text_meta: dict[str, Any] | None = None

    if explicit_local_data:
        local_data_path, local_data_origin = _resolve_local_data_path(
            explicit_local_data,
            training_root=training_root,
            prefer_training_root_data=prefer_training_root_data,
        )
        if not local_data_path.exists() or not local_data_path.is_file():
            raise SystemExit(f"Local training data not found: {local_data_path}")
        local_data_display = str(local_data_path)
    elif source_mode == "local":
        local_data_path, local_data_origin = _resolve_local_data_path(
            "",
            training_root=training_root,
            prefer_training_root_data=prefer_training_root_data,
        )
        if not local_data_path.exists() or not local_data_path.is_file():
            raise SystemExit(f"Local training data not found: {local_data_path}")
        local_data_display = str(local_data_path)
    else:
        stream_cfg = StreamDataConfig(
            dataset=str(args.stream_dataset or "").strip() or "HuggingFaceFW/fineweb-edu",
            config=(str(args.stream_config or "").strip() or None),
            split=str(args.stream_split or "train").strip() or "train",
            text_field=str(args.stream_field or "text").strip() or "text",
            row_filter_expr=str(args.stream_filter or "").strip(),
            max_chars=max(256, int(args.stream_max_chars)),
            min_chars=max(120, int(args.stream_min_chars)),
        )
        local_data_origin = "stream_primary"
        local_data_display = _stream_descriptor(stream_cfg)

    identity = _load_identity(identity_path)

    if not identity.get("client_id") or not identity.get("client_secret"):
        registration = _register(server_url, platform=args.platform, device_name=args.device_name, verify_tls=verify_tls)
        identity = {
            "client_id": registration["client_id"],
            "client_secret": registration["client_secret"],
            "platform": args.platform,
            "device_name": args.device_name,
        }
        _save_identity(identity_path, identity)
        print(f"Registered new client: {identity['client_id']}")

    round_payload = _fetch_round(server_url, identity["client_id"], identity["client_secret"], verify_tls=verify_tls)
    round_id = int(round_payload["round_id"])
    expected_base_sha = str(round_payload.get("base_model_sha256", "")).strip().lower()
    if expected_base_sha:
        base_model_path = Path(args.base_model).resolve()
        if not base_model_path.exists() or not base_model_path.is_file():
            raise SystemExit(f"Base model file not found for hash verification: {base_model_path}")
        actual_base_sha = file_sha256(str(base_model_path)).strip().lower()
        if actual_base_sha != expected_base_sha and not bool(args.allow_base_model_mismatch):
            raise SystemExit(
                "Server/local base model mismatch. "
                f"server={expected_base_sha} local={actual_base_sha} file={base_model_path}. "
                "Use --allow-base-model-mismatch to bypass (unsafe)."
            )
    lora_cfg = LoRAConfig.from_dict(round_payload["lora_config"])
    global_adapter = decode_tensor_dict(round_payload["global_adapter"])
    diloco_steps = int(round_payload.get("diloco_local_steps", 100))

    local_steps = diloco_steps if args.steps is None else args.steps
    train_cfg = LocalTrainConfig(
        steps=local_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        torch_threads=args.torch_threads,
        max_seconds=max(0.0, float(args.burst_seconds)),
        min_steps=max(1, int(args.min_steps)),
    )
    if stream_cfg is not None:
        try:
            training_text, training_text_meta = load_training_text(local_data_path="", stream_data=stream_cfg)
            delta, metrics = train_local_delta(
                base_model_path=args.base_model,
                lora_cfg=lora_cfg,
                global_adapter_state=global_adapter,
                local_data_path="",
                stream_data=stream_cfg,
                train_text=training_text,
                train_text_meta=training_text_meta,
                train_cfg=train_cfg,
            )
        except Exception as exc:
            if source_mode == "stream":
                raise SystemExit(f"Streaming source failed: {exc}") from None
            fallback_path, fallback_origin = _resolve_local_data_path(
                "",
                training_root=training_root,
                prefer_training_root_data=prefer_training_root_data,
            )
            if not fallback_path.exists() or not fallback_path.is_file():
                raise SystemExit(
                    "Streaming source failed and no local fallback corpus was found. "
                    f"stream={local_data_display} fallback={fallback_path} error={exc}"
                ) from None
            print(f"streaming unavailable; falling back to local corpus: {fallback_path}")
            local_data_path = fallback_path
            local_data_origin = f"{fallback_origin}_after_stream_error"
            local_data_display = str(local_data_path)
            stream_cfg = None
            training_text = ""
            training_text_meta = None
            delta, metrics = train_local_delta(
                base_model_path=args.base_model,
                lora_cfg=lora_cfg,
                global_adapter_state=global_adapter,
                local_data_path=str(local_data_path),
                train_cfg=train_cfg,
            )
    else:
        if local_data_path is None:
            raise SystemExit("No training data source resolved.")
        delta, metrics = train_local_delta(
            base_model_path=args.base_model,
            lora_cfg=lora_cfg,
            global_adapter_state=global_adapter,
            local_data_path=str(local_data_path),
            train_cfg=train_cfg,
        )

    local_eval_before = None
    local_eval_after = None
    local_loss_improvement = None
    gate_threshold = float(args.min_loss_improvement)
    if bool(args.require_local_improvement) and gate_threshold <= 0.0:
        gate_threshold = 1e-6
    gate_enabled = bool(args.require_local_improvement or gate_threshold > 0.0)
    gate_passed = True
    gate_reason = ""

    if gate_enabled:
        eval_data_path: Path | None = None
        eval_text = ""
        explicit_eval = str(args.local_eval_data or "").strip()
        if explicit_eval:
            eval_data_path = Path(explicit_eval).resolve()
            if not eval_data_path.exists() or not eval_data_path.is_file():
                raise SystemExit(f"Local eval data not found: {eval_data_path}")
        elif local_data_path is not None:
            eval_data_path = _resolve_eval_data_path("", local_data_path=local_data_path)
        elif training_text.strip():
            eval_text = training_text
        elif stream_cfg is not None:
            eval_text, _ = load_training_text(local_data_path="", stream_data=stream_cfg)

        eval_iters = max(1, int(args.local_eval_iters))
        local_eval_before = float(
            evaluate_adapter_loss(
                base_model_path=args.base_model,
                lora_cfg=lora_cfg,
                adapter_state=global_adapter,
                eval_data_path=str(eval_data_path) if eval_data_path is not None else "",
                eval_text=eval_text,
                eval_iters=eval_iters,
                batch_size=max(1, int(args.batch_size)),
                torch_threads=max(1, int(args.torch_threads)),
            )
        )
        candidate_state = add_states(global_adapter, delta)
        local_eval_after = float(
            evaluate_adapter_loss(
                base_model_path=args.base_model,
                lora_cfg=lora_cfg,
                adapter_state=candidate_state,
                eval_data_path=str(eval_data_path) if eval_data_path is not None else "",
                eval_text=eval_text,
                eval_iters=eval_iters,
                batch_size=max(1, int(args.batch_size)),
                torch_threads=max(1, int(args.torch_threads)),
            )
        )
        local_loss_improvement = float(local_eval_before - local_eval_after)
        gate_passed = local_loss_improvement >= gate_threshold
        if not gate_passed:
            gate_reason = (
                "local_eval_improvement_below_threshold "
                f"(improvement={local_loss_improvement:.6f}, required={gate_threshold:.6f})"
            )

    if local_eval_before is not None:
        metrics["local_eval_before"] = local_eval_before
    if local_eval_after is not None:
        metrics["local_eval_after"] = local_eval_after
    if local_loss_improvement is not None:
        metrics["local_eval_improvement"] = local_loss_improvement
    if gate_enabled:
        metrics["local_eval_required_improvement"] = gate_threshold
        metrics["local_eval_passed"] = bool(gate_passed)

    if not gate_passed:
        skipped = {
            "submitted": False,
            "reason": gate_reason,
            "round_id": round_id,
            "client_id": identity["client_id"],
            "local_data_path": local_data_display,
            "local_data_origin": local_data_origin,
            "metrics": metrics,
        }
        if args.json:
            print(json.dumps(skipped, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(skipped, indent=2, ensure_ascii=False))
        return

    envelope = {
        "client_id": identity["client_id"],
        "round_id": round_id,
        "num_examples": int(metrics.get("token_count", 1)),
        "metrics": metrics,
        "delta": encode_tensor_dict(delta),
        "timestamp": now_epoch_seconds(),
        "nonce": generate_nonce(),
    }
    envelope["signature"] = sign_payload(identity["client_secret"], envelope)
    result = _http_json("POST", f"{server_url}/v1/submit", envelope, verify_tls=verify_tls)
    # _http_json raises on any HTTP error, so reaching here means the
    # coordinator genuinely accepted this round's contribution -- this call
    # previously had zero callers anywhere in the codebase (confirmed via
    # repo-wide grep), so the seed:peer ratio shown in the UI could never
    # include training-round credit even though the ledger schema and
    # LedgerStore.record_training_round() already supported it.
    try:
        LedgerStore(DEFAULT_LEDGER_PATH).record_training_round()
    except Exception as exc:  # noqa: BLE001 -- the round already succeeded; don't fail the run over local bookkeeping
        print(f"Warning: could not record training round to contribution ledger: {exc}")
    result["submitted"] = True
    result["local_data_path"] = local_data_display
    result["local_data_origin"] = local_data_origin
    result["burst_mode"] = bool(float(args.burst_seconds) > 0.0)
    result["metrics"] = metrics

    if args.auto_torickle:
        try:
            delta_path = Path(f"data/federated/torickle_delta_{round_id}_{uuid.uuid4().hex[:8]}.pt")
            delta_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"delta": delta}, str(delta_path))
            bundle_out = Path(f"data/torickle/bundles/round_{round_id}")
            pack_result = pack_delta_file(
                delta_path=str(delta_path),
                out_dir=str(bundle_out),
                overwrite=True,
                metadata={"round_id": str(round_id), "model_hash": args.model_hash},
            )
            swarm_identity = ensure_identity(DEFAULT_IDENTITY_PATH, label="federated-client")
            swarm = SwarmNode(identity=swarm_identity, data_dir=DEFAULT_DATA_DIR)
            swarm.import_bundle(str(bundle_out))
            bundle_ids = list(swarm.bundles.keys())
            if bundle_ids:
                ann = swarm.announce_bundle(bundle_ids[0], model_hash=args.model_hash)
                if ann:
                    result["torickle_bundle_id"] = bundle_ids[0]
                    result["torickle_announced"] = True
            delta_path.unlink(missing_ok=True)
        except Exception as exc:
            result["torickle_error"] = str(exc)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
