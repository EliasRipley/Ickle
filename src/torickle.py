from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.federated.protocol import decode_tensor_dict, encode_tensor_dict


class TorickleError(RuntimeError):
    pass


@dataclass
class PieceRecord:
    index: int
    offset_bytes: int
    size_bytes: int
    sha256: str
    leaf_hash: str
    file_name: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leaf_hash_hex(piece_bytes: bytes) -> str:
    return hashlib.sha256(b"\x00" + piece_bytes).hexdigest()


def _parent_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _merkle_root_hex(leaf_hashes: list[str]) -> str:
    if not leaf_hashes:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(value) for value in leaf_hashes]
    while len(level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_parent_hash(left, right))
        level = next_level
    return level[0].hex()


def _extract_delta_from_payload(payload: Any) -> dict[str, torch.Tensor]:
    candidate = payload
    if isinstance(payload, dict) and isinstance(payload.get("delta"), dict):
        candidate = payload["delta"]
    if not isinstance(candidate, dict):
        raise TorickleError("Delta payload must be a tensor dict or an object containing a 'delta' tensor dict.")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(value, torch.Tensor):
            raise TorickleError("Delta payload contains non-tensor values.")
        normalized[str(key)] = value.detach().cpu().float().contiguous()
    return normalized


def _delta_to_payload_bytes(delta: dict[str, torch.Tensor]) -> bytes:
    wire = encode_tensor_dict(delta)
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _payload_bytes_to_delta(payload_bytes: bytes) -> dict[str, torch.Tensor]:
    try:
        wire = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TorickleError(f"Failed to decode Torickle payload JSON: {exc}") from exc
    if not isinstance(wire, dict):
        raise TorickleError("Torickle payload must decode to a JSON object.")
    return decode_tensor_dict(wire)


def _parse_meta(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise TorickleError(f"Invalid --meta entry '{item}'. Expected format key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise TorickleError("Invalid --meta entry with empty key.")
        out[key] = value
    return out


def pack_delta_file(
    *,
    delta_path: str,
    out_dir: str,
    piece_size_bytes: int = 256 * 1024,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if piece_size_bytes < 64:
        raise TorickleError("piece_size_bytes must be at least 64.")

    source = Path(delta_path).resolve()
    if not source.exists() or not source.is_file():
        raise TorickleError(f"Delta file not found: {source}")

    payload = torch.load(source, map_location="cpu")
    delta = _extract_delta_from_payload(payload)
    payload_bytes = _delta_to_payload_bytes(delta)

    root = Path(out_dir).resolve()
    pieces_dir = root / "pieces"
    if root.exists():
        if not overwrite:
            raise TorickleError(f"Output directory already exists: {root}. Use --overwrite to replace it.")
        shutil.rmtree(root)
    pieces_dir.mkdir(parents=True, exist_ok=True)

    payload_hash = hashlib.sha256()
    piece_records: list[PieceRecord] = []
    leaf_hashes: list[str] = []
    offset = 0
    piece_index = 0
    while offset < len(payload_bytes):
        piece = payload_bytes[offset : offset + piece_size_bytes]
        piece_sha = _sha256_bytes(piece)
        leaf_hash = _leaf_hash_hex(piece)
        file_name = f"piece_{piece_index:06d}_{piece_sha[:16]}.bin"
        target = pieces_dir / file_name
        target.write_bytes(piece)
        payload_hash.update(piece)
        piece_records.append(
            PieceRecord(
                index=piece_index,
                offset_bytes=offset,
                size_bytes=len(piece),
                sha256=piece_sha,
                leaf_hash=leaf_hash,
                file_name=file_name,
            )
        )
        leaf_hashes.append(leaf_hash)
        piece_index += 1
        offset += len(piece)

    manifest = {
        "torickle_version": "0.1",
        "created_at_utc": _utc_now(),
        "payload_format": "lora_delta_tensor_dict_wire_v1",
        "payload_sha256": payload_hash.hexdigest(),
        "merkle_root": _merkle_root_hex(leaf_hashes),
        "piece_size_bytes": int(piece_size_bytes),
        "piece_count": len(piece_records),
        "total_bytes": len(payload_bytes),
        "pieces_dir": "pieces",
        "pieces": [asdict(piece) for piece in piece_records],
        "metadata": {
            "source_file": str(source),
            **(metadata or {}),
        },
    }

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "out_dir": str(root),
        "piece_count": len(piece_records),
        "total_bytes": len(payload_bytes),
        "payload_sha256": manifest["payload_sha256"],
        "merkle_root": manifest["merkle_root"],
    }


def verify_manifest(*, manifest_path: str) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.exists() or not manifest_file.is_file():
        return {
            "valid": False,
            "manifest_path": str(manifest_file),
            "errors": [f"Manifest not found: {manifest_file}"],
            "warnings": [],
        }

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "manifest_path": str(manifest_file),
            "errors": [f"Failed to parse manifest JSON: {exc}"],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []

    pieces_dir = manifest_file.parent / str(manifest.get("pieces_dir", "pieces"))
    pieces = manifest.get("pieces", [])
    if not isinstance(pieces, list):
        errors.append("manifest.pieces must be a list.")
        pieces = []

    payload_hash = hashlib.sha256()
    leaf_hashes: list[str] = []
    total_bytes = 0

    for idx, row in enumerate(pieces):
        if not isinstance(row, dict):
            errors.append(f"piece[{idx}] is not an object.")
            continue
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            errors.append(f"piece[{idx}] missing file_name.")
            continue
        path = pieces_dir / file_name
        if not path.exists() or not path.is_file():
            errors.append(f"piece[{idx}] file missing: {path}")
            continue
        data = path.read_bytes()
        total_bytes += len(data)
        payload_hash.update(data)
        piece_sha = _sha256_bytes(data)
        leaf_hash = _leaf_hash_hex(data)
        leaf_hashes.append(leaf_hash)

        expected_size = int(row.get("size_bytes", -1))
        if expected_size >= 0 and expected_size != len(data):
            errors.append(f"piece[{idx}] size mismatch ({len(data)} != {expected_size}).")
        expected_sha = str(row.get("sha256", "")).lower()
        if expected_sha and expected_sha != piece_sha:
            errors.append(f"piece[{idx}] sha256 mismatch.")
        expected_leaf = str(row.get("leaf_hash", "")).lower()
        if expected_leaf and expected_leaf != leaf_hash:
            errors.append(f"piece[{idx}] leaf_hash mismatch.")
        if int(row.get("index", idx)) != idx:
            warnings.append(f"piece[{idx}] index field does not match list order.")

    expected_payload_sha = str(manifest.get("payload_sha256", "")).lower()
    actual_payload_sha = payload_hash.hexdigest()
    if expected_payload_sha and expected_payload_sha != actual_payload_sha:
        errors.append("payload_sha256 mismatch.")

    expected_total_bytes = int(manifest.get("total_bytes", -1))
    if expected_total_bytes >= 0 and expected_total_bytes != total_bytes:
        errors.append(f"total_bytes mismatch ({total_bytes} != {expected_total_bytes}).")

    expected_piece_count = int(manifest.get("piece_count", -1))
    if expected_piece_count >= 0 and expected_piece_count != len(pieces):
        errors.append(f"piece_count mismatch ({len(pieces)} != {expected_piece_count}).")

    expected_root = str(manifest.get("merkle_root", "")).lower()
    actual_root = _merkle_root_hex(leaf_hashes)
    if expected_root and expected_root != actual_root:
        errors.append("merkle_root mismatch.")

    return {
        "valid": not errors,
        "manifest_path": str(manifest_file),
        "errors": errors,
        "warnings": warnings,
        "piece_count": len(pieces),
        "total_bytes": total_bytes,
        "payload_sha256": actual_payload_sha,
        "merkle_root": actual_root,
    }


def reassemble_manifest(
    *,
    manifest_path: str,
    out_path: str,
    strict_verify: bool = True,
) -> dict[str, Any]:
    report = verify_manifest(manifest_path=manifest_path)
    if strict_verify and not report.get("valid", False):
        raise TorickleError("Manifest verification failed: " + "; ".join(report.get("errors", [])))

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    pieces_dir = manifest_file.parent / str(manifest.get("pieces_dir", "pieces"))
    pieces = manifest.get("pieces", [])
    if not isinstance(pieces, list):
        raise TorickleError("manifest.pieces must be a list.")

    payload_parts: list[bytes] = []
    for idx, row in enumerate(pieces):
        if not isinstance(row, dict):
            raise TorickleError(f"piece[{idx}] is not an object.")
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            raise TorickleError(f"piece[{idx}] missing file_name.")
        path = pieces_dir / file_name
        if not path.exists() or not path.is_file():
            raise TorickleError(f"piece[{idx}] file missing: {path}")
        payload_parts.append(path.read_bytes())

    payload_bytes = b"".join(payload_parts)
    delta = _payload_bytes_to_delta(payload_bytes)

    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, out_file)
    return {
        "out_path": str(out_file),
        "tensor_count": len(delta),
        "payload_sha256": _sha256_bytes(payload_bytes),
        "verified": bool(report.get("valid", False)),
    }


def _print_payload(payload: Any, as_json: bool):
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    print(payload)


def main():
    parser = argparse.ArgumentParser(description="Torickle v0: piece/verify/reassemble transport format for federated deltas.")
    sub = parser.add_subparsers(dest="command", required=True)

    pack_parser = sub.add_parser("pack", help="Pack a delta tensor dict file into Torickle pieces + manifest.")
    pack_parser.add_argument("--delta", required=True, help="Path to .pt file containing delta tensor dict (or update file with delta field).")
    pack_parser.add_argument("--out-dir", required=True, help="Output directory for manifest and piece files.")
    pack_parser.add_argument("--piece-size-bytes", type=int, default=256 * 1024, help="Target piece size in bytes.")
    pack_parser.add_argument("--overwrite", action="store_true", help="Replace out-dir if it already exists.")
    pack_parser.add_argument("--meta", action="append", default=[], help="Optional metadata key=value (repeatable).")
    pack_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    verify_parser = sub.add_parser("verify", help="Verify Torickle piece files against manifest.")
    verify_parser.add_argument("--manifest", required=True, help="Path to Torickle manifest.json.")
    verify_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    reassemble_parser = sub.add_parser("reassemble", help="Reassemble Torickle pieces into a delta tensor dict .pt file.")
    reassemble_parser.add_argument("--manifest", required=True, help="Path to Torickle manifest.json.")
    reassemble_parser.add_argument("--out", required=True, help="Destination .pt file for reconstructed delta tensor dict.")
    reassemble_parser.add_argument(
        "--no-strict-verify",
        action="store_true",
        help="Allow reassembly even if verification reports errors.",
    )
    reassemble_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    args = parser.parse_args()
    as_json = bool(getattr(args, "json", False))
    try:
        if args.command == "pack":
            result = pack_delta_file(
                delta_path=args.delta,
                out_dir=args.out_dir,
                piece_size_bytes=int(args.piece_size_bytes),
                overwrite=bool(args.overwrite),
                metadata=_parse_meta(list(args.meta or [])),
            )
            _print_payload(result, as_json)
            return
        if args.command == "verify":
            result = verify_manifest(manifest_path=args.manifest)
            _print_payload(result, as_json)
            if not result.get("valid", False):
                raise SystemExit(1)
            return
        if args.command == "reassemble":
            result = reassemble_manifest(
                manifest_path=args.manifest,
                out_path=args.out,
                strict_verify=not bool(args.no_strict_verify),
            )
            _print_payload(result, as_json)
            return
        raise TorickleError(f"Unknown command: {args.command}")
    except TorickleError as exc:
        if as_json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
