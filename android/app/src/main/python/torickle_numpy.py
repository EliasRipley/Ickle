import hashlib
import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from numpy_wire import encode_tensor_dict, decode_tensor_dict


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _leaf_hash_hex(piece_bytes):
    return hashlib.sha256(b"\x00" + piece_bytes).hexdigest()


def _parent_hash(left, right):
    return hashlib.sha256(b"\x01" + left + right).digest()


def _merkle_root_hex(leaf_hashes):
    if not leaf_hashes:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(h) for h in leaf_hashes]
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_parent_hash(left, right))
        level = next_level
    return level[0].hex()


def _delta_to_payload_bytes(delta):
    wire = encode_tensor_dict(delta)
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _payload_bytes_to_delta(payload_bytes):
    try:
        wire = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to decode Torickle payload JSON: {exc}")
    if not isinstance(wire, dict):
        raise RuntimeError("Torickle payload must decode to a JSON object.")
    return decode_tensor_dict(wire)


def pack_delta(delta, out_dir, piece_size_bytes=256 * 1024, overwrite=False, metadata=None):
    if piece_size_bytes < 64:
        raise ValueError("piece_size_bytes must be at least 64.")
    if not isinstance(delta, dict):
        raise ValueError("delta must be a dict of name -> numpy array.")

    payload_bytes = _delta_to_payload_bytes(delta)

    root = Path(out_dir).resolve()
    pieces_dir = root / "pieces"
    if root.exists():
        if not overwrite:
            raise RuntimeError(f"Output directory already exists: {root}. Use overwrite=True to replace it.")
        shutil.rmtree(str(root))
    pieces_dir.mkdir(parents=True, exist_ok=True)

    payload_hash = hashlib.sha256()
    piece_records = []
    leaf_hashes = []
    offset = 0
    piece_index = 0
    while offset < len(payload_bytes):
        piece = payload_bytes[offset:offset + piece_size_bytes]
        piece_sha = _sha256_bytes(piece)
        leaf_hash = _leaf_hash_hex(piece)
        file_name = f"piece_{piece_index:06d}_{piece_sha[:16]}.bin"
        target = pieces_dir / file_name
        target.write_bytes(piece)
        payload_hash.update(piece)
        piece_records.append({
            "index": piece_index,
            "offset_bytes": offset,
            "size_bytes": len(piece),
            "sha256": piece_sha,
            "leaf_hash": leaf_hash,
            "file_name": file_name,
        })
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
        "pieces": piece_records,
        "metadata": metadata or {},
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


def verify_manifest(manifest_path):
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
    except Exception as exc:
        return {
            "valid": False,
            "manifest_path": str(manifest_file),
            "errors": [f"Failed to parse manifest JSON: {exc}"],
            "warnings": [],
        }

    errors = []
    warnings = []

    pieces_dir = manifest_file.parent / str(manifest.get("pieces_dir", "pieces"))
    pieces = manifest.get("pieces", [])
    if not isinstance(pieces, list):
        errors.append("manifest.pieces must be a list.")
        pieces = []

    payload_hash = hashlib.sha256()
    leaf_hashes = []
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


def reassemble_manifest(manifest_path, out_path, strict_verify=True):
    report = verify_manifest(manifest_path)
    if strict_verify and not report.get("valid", False):
        raise RuntimeError("Manifest verification failed: " + "; ".join(report.get("errors", [])))

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    pieces_dir = manifest_file.parent / str(manifest.get("pieces_dir", "pieces"))
    pieces = manifest.get("pieces", [])
    if not isinstance(pieces, list):
        raise RuntimeError("manifest.pieces must be a list.")

    payload_parts = []
    for idx, row in enumerate(pieces):
        if not isinstance(row, dict):
            raise RuntimeError(f"piece[{idx}] is not an object.")
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            raise RuntimeError(f"piece[{idx}] missing file_name.")
        path = pieces_dir / file_name
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"piece[{idx}] file missing: {path}")
        payload_parts.append(path.read_bytes())

    payload_bytes = b"".join(payload_parts)
    delta = _payload_bytes_to_delta(payload_bytes)

    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_file.with_suffix("")), **delta)
    return {
        "out_path": str(out_file),
        "tensor_count": len(delta),
        "payload_sha256": _sha256_bytes(payload_bytes),
        "verified": bool(report.get("valid", False)),
    }
