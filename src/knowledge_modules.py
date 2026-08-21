from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from src.workspace_paths import get_training_root


@dataclass
class ResolvedKnowledgeModule:
    module_id: str
    path: str
    weight: float = 1.0
    rank: int | None = None
    alpha: float | None = None
    score: float = 0.0
    topics: list[str] = field(default_factory=list)
    description: str = ""


def default_registry_path() -> str:
    return str(get_training_root() / "knowledge" / "module_registry.json")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return int(default)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9']+", str(text or "").lower()) if len(tok) >= 3}


def _default_registry() -> dict[str, Any]:
    return {"version": 1, "modules": []}


def load_registry(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return _default_registry()
    try:
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return _default_registry()
    if not isinstance(payload, dict):
        return _default_registry()
    modules = payload.get("modules")
    if not isinstance(modules, list):
        payload["modules"] = []
    payload.setdefault("version", 1)
    return payload


def save_registry(path: str, payload: dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def register_module(
    *,
    registry_path: str,
    module_id: str,
    module_path: str,
    topics: list[str] | None = None,
    description: str = "",
    base_model: str = "",
    weight: float = 1.0,
    rank: int = 0,
    alpha: float = 0.0,
    priority: float = 0.0,
    enabled: bool = True,
    default_active: bool = False,
) -> dict[str, Any]:
    module_id = str(module_id or "").strip()
    if not module_id:
        raise ValueError("module_id is required")

    module_path = str(module_path or "").strip()
    if not module_path:
        raise ValueError("module_path is required")
    if not Path(module_path).exists():
        raise FileNotFoundError(f"module_path does not exist: {module_path}")

    registry = load_registry(registry_path)
    modules = [row for row in list(registry.get("modules", [])) if isinstance(row, dict)]

    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()]
    row: dict[str, Any] = {
        "id": module_id,
        "path": module_path,
        "topics": topic_list,
        "description": str(description or "").strip(),
        "base_model": str(base_model or "").strip(),
        "weight": float(weight),
        "rank": int(rank) if int(rank) > 0 else None,
        "alpha": float(alpha) if float(alpha) > 0 else None,
        "priority": float(priority),
        "enabled": bool(enabled),
        "default_active": bool(default_active),
    }

    replaced = False
    for idx, existing in enumerate(modules):
        if str(existing.get("id", "")).strip() == module_id:
            modules[idx] = row
            replaced = True
            break
    if not replaced:
        modules.append(row)

    registry["modules"] = modules
    save_registry(registry_path, registry)
    return {"registered": True, "replaced": replaced, "module": row, "registry_path": registry_path}


def parse_module_ids(value: str) -> list[str]:
    out: list[str] = []
    for piece in str(value or "").split(","):
        module_id = piece.strip()
        if module_id:
            out.append(module_id)
    return out


def _base_model_matches(module_base_model: str, requested_base_model: str) -> bool:
    module_base_model = str(module_base_model or "").strip()
    if not module_base_model:
        return True
    requested_base_model = str(requested_base_model or "").strip()
    if not requested_base_model:
        return True
    module_path = Path(module_base_model)
    requested_path = Path(requested_base_model)
    try:
        return module_path.resolve() == requested_path.resolve()
    except Exception:  # noqa: BLE001
        return module_path.name.lower() == requested_path.name.lower()


def _module_score(prompt: str, module_row: dict[str, Any]) -> float:
    prompt_lower = str(prompt or "").lower()
    prompt_tokens = _tokenize(prompt_lower)
    if not prompt_tokens:
        return 0.0

    topics = [str(t).strip().lower() for t in list(module_row.get("topics") or []) if str(t).strip()]
    description = str(module_row.get("description", "")).strip().lower()
    text = " ".join([str(module_row.get("id", ""))] + topics + [description])
    module_tokens = _tokenize(text)
    overlap = len(prompt_tokens.intersection(module_tokens)) / max(1, len(prompt_tokens))

    phrase_bonus = 0.0
    for topic in topics:
        if len(topic) >= 4 and topic in prompt_lower:
            phrase_bonus += 0.35

    priority = _safe_float(module_row.get("priority", 0.0), 0.0)
    default_bonus = 0.05 if bool(module_row.get("default_active", False)) else 0.0
    return float(overlap + phrase_bonus + (0.02 * priority) + default_bonus)


def resolve_runtime_modules(
    *,
    prompt: str,
    base_model: str,
    registry_path: str = default_registry_path(),
    explicit_module_ids: list[str] | None = None,
    max_modules: int = 2,
    auto_select: bool = True,
) -> list[ResolvedKnowledgeModule]:
    registry = load_registry(registry_path)
    rows = [row for row in list(registry.get("modules", [])) if isinstance(row, dict)]

    eligible: dict[str, dict[str, Any]] = {}
    for row in rows:
        module_id = str(row.get("id", "")).strip()
        if not module_id:
            continue
        if not bool(row.get("enabled", True)):
            continue
        module_path = str(row.get("path", "")).strip()
        if not module_path or not Path(module_path).exists():
            continue
        if not _base_model_matches(str(row.get("base_model", "")), base_model):
            continue
        eligible[module_id] = row

    resolved: list[ResolvedKnowledgeModule] = []
    explicit_ids = [m for m in (explicit_module_ids or []) if m]
    if explicit_ids:
        for module_id in explicit_ids:
            row = eligible.get(module_id)
            if not row:
                continue
            resolved.append(
                ResolvedKnowledgeModule(
                    module_id=module_id,
                    path=str(row.get("path")),
                    weight=_safe_float(row.get("weight", 1.0), 1.0),
                    rank=_safe_int(row.get("rank", 0), 0) or None,
                    alpha=_safe_float(row.get("alpha", 0.0), 0.0) or None,
                    score=1.0,
                    topics=[str(t) for t in list(row.get("topics") or []) if str(t).strip()],
                    description=str(row.get("description", "")),
                )
            )
            if len(resolved) >= max(1, int(max_modules)):
                break
        return resolved

    if not auto_select:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in eligible.values():
        score = _module_score(prompt, row)
        if score <= 0:
            continue
        scored.append((score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    for score, row in scored[: max(1, int(max_modules))]:
        resolved.append(
            ResolvedKnowledgeModule(
                module_id=str(row.get("id")),
                path=str(row.get("path")),
                weight=_safe_float(row.get("weight", 1.0), 1.0),
                rank=_safe_int(row.get("rank", 0), 0) or None,
                alpha=_safe_float(row.get("alpha", 0.0), 0.0) or None,
                score=float(score),
                topics=[str(t) for t in list(row.get("topics") or []) if str(t).strip()],
                description=str(row.get("description", "")),
            )
        )
    return resolved


def _extract_lora_state(module_path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(module_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"module payload is not a dict: {module_path}")

    if "lora_state" in payload and isinstance(payload["lora_state"], dict):
        meta: dict[str, Any] = {}
        if isinstance(payload.get("lora"), dict):
            meta.update(payload["lora"])
        if "lora_rank" in payload:
            meta["rank"] = payload["lora_rank"]
        if "lora_alpha" in payload:
            meta["alpha"] = payload["lora_alpha"]
        return payload["lora_state"], meta

    tensor_keys = [k for k, v in payload.items() if isinstance(k, str) and isinstance(v, torch.Tensor)]
    if tensor_keys and all(k.endswith(".lora_a") or k.endswith(".lora_b") for k in tensor_keys):
        state = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        return state, {}

    raise ValueError(
        "module payload must contain 'lora_state' or be a raw LoRA tensor dict"
    )


def _collect_adapter_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            continue
        if key.endswith(".lora_a"):
            name = key[: -len(".lora_a")]
            pairs.setdefault(name, {})["a"] = tensor
        elif key.endswith(".lora_b"):
            name = key[: -len(".lora_b")]
            pairs.setdefault(name, {})["b"] = tensor
    return pairs


def _infer_rank(state: dict[str, torch.Tensor]) -> int:
    for key, tensor in state.items():
        if key.endswith(".lora_a") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
    return 0


def apply_lora_modules_to_model(model: nn.Module, modules: list[ResolvedKnowledgeModule]) -> dict[str, Any]:
    module_index = dict(model.named_modules())
    report: dict[str, Any] = {
        "modules_requested": len(modules),
        "modules_applied": 0,
        "applied_layers": 0,
        "module_reports": [],
    }

    for entry in modules:
        module_report: dict[str, Any] = {
            "module_id": entry.module_id,
            "path": entry.path,
            "applied_layers": 0,
            "weight": float(entry.weight),
        }
        try:
            state, meta = _extract_lora_state(entry.path)
        except Exception as exc:  # noqa: BLE001
            module_report["error"] = str(exc)
            report["module_reports"].append(module_report)
            continue

        rank = int(entry.rank or 0)
        if rank <= 0:
            rank = _safe_int(meta.get("rank", 0), 0)
        if rank <= 0:
            rank = _infer_rank(state)
        if rank <= 0:
            module_report["error"] = "could not infer rank for module"
            report["module_reports"].append(module_report)
            continue

        alpha = float(entry.alpha or 0.0)
        if alpha <= 0:
            alpha = _safe_float(meta.get("alpha", 0.0), 0.0)
        if alpha <= 0:
            alpha = float(rank)
        scale = alpha / max(1, rank)

        pairs = _collect_adapter_pairs(state)
        for layer_name, tensors in pairs.items():
            a = tensors.get("a")
            b = tensors.get("b")
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                continue

            target = module_index.get(layer_name)
            if not isinstance(target, nn.Linear):
                continue
            if a.ndim != 2 or b.ndim != 2:
                continue

            delta = (b.float() @ a.float()) * scale * float(entry.weight)
            if target.weight.shape != delta.shape:
                continue

            target.weight.data.add_(delta.to(device=target.weight.device, dtype=target.weight.dtype))
            module_report["applied_layers"] += 1
            report["applied_layers"] += 1

        if module_report["applied_layers"] > 0:
            report["modules_applied"] += 1
        report["module_reports"].append(module_report)

    return report


def _resolved_to_dict(modules: list[ResolvedKnowledgeModule]) -> list[dict[str, Any]]:
    return [asdict(item) for item in modules]


def _cli_list(args):
    registry = load_registry(args.registry)
    rows = [row for row in list(registry.get("modules", [])) if isinstance(row, dict)]
    if args.json:
        print(json.dumps({"registry_path": args.registry, "count": len(rows), "modules": rows}, indent=2, ensure_ascii=False))
        return
    print(f"registry={args.registry}")
    for row in rows:
        topics = ",".join([str(t) for t in list(row.get("topics") or []) if str(t).strip()])
        print(
            f"- {row.get('id')} path={row.get('path')} enabled={row.get('enabled', True)} "
            f"weight={row.get('weight', 1.0)} topics={topics}"
        )


def _cli_register(args):
    result = register_module(
        registry_path=args.registry,
        module_id=args.module_id,
        module_path=args.module_path,
        topics=[x.strip() for x in str(args.topics or "").split(",") if x.strip()],
        description=args.description,
        base_model=args.base_model,
        weight=float(args.weight),
        rank=int(args.rank),
        alpha=float(args.alpha),
        priority=float(args.priority),
        enabled=bool(args.enabled),
        default_active=bool(args.default_active),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else f"registered={result['module']['id']}")


def _cli_resolve(args):
    resolved = resolve_runtime_modules(
        prompt=args.prompt,
        base_model=args.base_model,
        registry_path=args.registry,
        explicit_module_ids=parse_module_ids(args.module_ids),
        max_modules=max(1, int(args.max_modules)),
        auto_select=bool(args.auto_select),
    )
    payload = {
        "prompt": args.prompt,
        "base_model": args.base_model,
        "registry_path": args.registry,
        "count": len(resolved),
        "modules": _resolved_to_dict(resolved),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else payload)


def main():
    parser = argparse.ArgumentParser(description="Knowledge module registry and runtime resolver.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--registry", default=default_registry_path())
    p_list.add_argument("--json", action="store_true")

    p_register = sub.add_parser("register")
    p_register.add_argument("--registry", default=default_registry_path())
    p_register.add_argument("--module-id", required=True)
    p_register.add_argument("--module-path", required=True)
    p_register.add_argument("--topics", default="")
    p_register.add_argument("--description", default="")
    p_register.add_argument("--base-model", default="")
    p_register.add_argument("--weight", type=float, default=1.0)
    p_register.add_argument("--rank", type=int, default=0)
    p_register.add_argument("--alpha", type=float, default=0.0)
    p_register.add_argument("--priority", type=float, default=0.0)
    p_register.add_argument("--enabled", dest="enabled", action="store_true")
    p_register.add_argument("--disabled", dest="enabled", action="store_false")
    p_register.set_defaults(enabled=True)
    p_register.add_argument("--default-active", action="store_true")
    p_register.add_argument("--json", action="store_true")

    p_resolve = sub.add_parser("resolve")
    p_resolve.add_argument("--registry", default=default_registry_path())
    p_resolve.add_argument("--prompt", required=True)
    p_resolve.add_argument("--base-model", default="")
    p_resolve.add_argument("--module-ids", default="", help="Comma separated explicit module ids")
    p_resolve.add_argument("--max-modules", type=int, default=2)
    p_resolve.add_argument("--auto-select", dest="auto_select", action="store_true")
    p_resolve.add_argument("--no-auto-select", dest="auto_select", action="store_false")
    p_resolve.set_defaults(auto_select=True)
    p_resolve.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "list":
        _cli_list(args)
        return
    if args.command == "register":
        _cli_register(args)
        return
    if args.command == "resolve":
        _cli_resolve(args)
        return


if __name__ == "__main__":
    main()
