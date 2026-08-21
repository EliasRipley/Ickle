"""Model Knowledge Accumulation for Ickle.

Merges trained topic models into the master model using weighted averaging
with quality gates. Supports same-architecture merging (FedAvg / model soup)
and corpus-level accumulation for cross-architecture knowledge transfer.

Research basis:
  - Model Soup (Wortsman et al., 2022): averaging fine-tuned weights
  - DiLoCo (Douillard et al., 2022): dual-optimizer federated merging
  - EWC-DR (CVPR 2026): corrected Fisher importance for continual learning
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class AccumulationConfig:
    min_quality_score: float = 0.35
    master_weight: float = 0.85
    candidate_weight: float = 0.15
    max_regression: float = 0.03
    archive_candidate: bool = True
    archive_dir: str = "models/archive"


def _read_meta(model_path: str) -> dict[str, Any]:
    meta_path = Path(str(model_path) + ".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _architecture_key_from_config(cfg_dict: dict[str, Any]) -> str:
    keys = ["vocab_size", "block_size", "n_embd", "n_head", "n_layer"]
    parts = [f"{k}={cfg_dict.get(k, '?')}" for k in keys]
    return "|".join(parts)


def architectures_compatible(model_a: str, model_b: str) -> bool:
    try:
        ckpt_a = torch.load(model_a, map_location="cpu")
        ckpt_b = torch.load(model_b, map_location="cpu")
        cfg_a = ckpt_a.get("config", {})
        cfg_b = ckpt_b.get("config", {})
        return _architecture_key_from_config(cfg_a) == _architecture_key_from_config(cfg_b)
    except Exception:
        return False


def accumulate_weights(
    master_path: str,
    candidate_path: str,
    *,
    master_weight: float = 0.85,
    candidate_weight: float = 0.15,
) -> dict[str, torch.Tensor]:
    master_ckpt = torch.load(master_path, map_location="cpu")
    candidate_ckpt = torch.load(candidate_path, map_location="cpu")

    master_state = master_ckpt.get("model_state", master_ckpt.get("model"))
    candidate_state = candidate_ckpt.get("model_state", candidate_ckpt.get("model"))

    if master_state is None:
        master_state = {k: v for k, v in master_ckpt.items() if isinstance(v, torch.Tensor)}
    if candidate_state is None:
        candidate_state = {k: v for k, v in candidate_ckpt.items() if isinstance(v, torch.Tensor)}

    merged: dict[str, torch.Tensor] = {}
    for key in master_state:
        m = master_state[key].float()
        c = candidate_state.get(key, m).float()
        merged[key] = m * master_weight + c * candidate_weight

    return merged


def save_accumulated(
    merged_state: dict[str, torch.Tensor],
    master_path: str,
    candidate_meta: dict[str, Any],
):
    master_ckpt = torch.load(master_path, map_location="cpu")
    master_state_key = None
    if "model_state" in master_ckpt:
        master_state_key = "model_state"
    elif "model" in master_ckpt:
        master_state_key = "model"

    if master_state_key:
        master_ckpt[master_state_key] = {k: v.clone() for k, v in merged_state.items()}
    else:
        for k in list(master_ckpt.keys()):
            if k in merged_state:
                master_ckpt[k] = merged_state[k]

    backup = Path(str(master_path) + ".backup.pt")
    if Path(master_path).exists():
        import shutil
        shutil.copyfile(master_path, backup)

    model_path = Path(master_path)
    temp_model = model_path.with_suffix(model_path.suffix + ".tmp")
    torch.save(master_ckpt, temp_model)
    os.replace(temp_model, model_path)

    master_meta = _read_meta(master_path)
    master_meta.setdefault("accumulation_history", [])
    master_meta["accumulation_history"].append({
        "candidate_meta": candidate_meta,
        "timestamp": "",
        "master_weight": 0.85,
        "candidate_weight": 0.15,
    })
    meta_path = Path(str(master_path) + ".meta.json")
    temp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    temp_meta.write_text(json.dumps(master_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_meta, meta_path)


def accumulate_corpus(
    candidate_corpus_path: str,
    master_corpus_path: str,
    *,
    max_lines: int = 100_000,
):
    """Append candidate training corpus to master corpus for cross-architecture knowledge transfer."""
    from pathlib import Path

    candidate = Path(candidate_corpus_path)
    if not candidate.exists():
        return

    master = Path(master_corpus_path)
    master.parent.mkdir(parents=True, exist_ok=True)

    candidate_lines = candidate.read_text(encoding="utf-8").splitlines()
    master_lines = existing_lines(master)
    existing: set[str] = set(master_lines)

    new_lines = [ln for ln in candidate_lines if ln.strip() and ln not in existing]
    all_lines = master_lines + new_lines

    if len(all_lines) > max_lines:
        all_lines = all_lines[-max_lines:]

    master.write_text("\n".join(all_lines) + "\n", encoding="utf-8")


def existing_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


class Accumulator:
    def __init__(self, config: AccumulationConfig | None = None):
        self.cfg = config or AccumulationConfig()

    def try_accumulate(
        self,
        master_path: str,
        candidate_path: str,
        candidate_score: float,
        *,
        candidate_corpus: str = "",
        master_corpus: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "master": master_path,
            "candidate": candidate_path,
            "score": candidate_score,
            "merged": False,
            "method": "none",
            "reason": "",
        }

        if candidate_score < self.cfg.min_quality_score:
            report["reason"] = f"Score {candidate_score:.3f} below min {self.cfg.min_quality_score}"
            return report

        compatible = architectures_compatible(master_path, candidate_path)

        if compatible:
            report["method"] = "weighted_average"
            if not dry_run:
                merged_state = accumulate_weights(
                    master_path,
                    candidate_path,
                    master_weight=self.cfg.master_weight,
                    candidate_weight=self.cfg.candidate_weight,
                )
                save_accumulated(merged_state, master_path, _read_meta(candidate_path))
                report["merged"] = True
            else:
                report["merged"] = True
                report["dry_run"] = True
        elif candidate_corpus and master_corpus:
            report["method"] = "corpus_accumulation"
            if not dry_run:
                accumulate_corpus(candidate_corpus, master_corpus)
                report["merged"] = True
                report["corpus_file"] = master_corpus
            else:
                report["merged"] = True
                report["dry_run"] = True
        else:
            report["reason"] = "Architectures incompatible and no corpus available"

        return report
