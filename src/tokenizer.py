from __future__ import annotations

import base64
import tempfile
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TokenizerError(RuntimeError):
    pass


class BaseTokenizer:
    kind: str = "base"

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def checkpoint_payload(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class CharTokenizer(BaseTokenizer):
    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "?"
    kind: str = "char"

    @staticmethod
    def from_text(text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        if "?" not in stoi:
            idx = len(stoi)
            stoi["?"] = idx
            itos[idx] = "?"
        return CharTokenizer(stoi=stoi, itos=itos, unk_token="?")

    @staticmethod
    def from_checkpoint(stoi: dict[str, int], itos: dict[int, str], unk_token: str = "?") -> "CharTokenizer":
        # Legacy checkpoints sometimes persist int-key dict as strings.
        normalized_itos: dict[int, str] = {}
        for key, value in itos.items():
            normalized_itos[int(key)] = str(value)
        normalized_stoi = {str(k): int(v) for k, v in stoi.items()}
        if unk_token not in normalized_stoi:
            idx = len(normalized_stoi)
            normalized_stoi[unk_token] = idx
            normalized_itos[idx] = unk_token
        return CharTokenizer(stoi=normalized_stoi, itos=normalized_itos, unk_token=unk_token)

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi.get(self.unk_token, next(iter(self.stoi.values())))
        return [int(self.stoi.get(ch, unknown)) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos.get(int(i), self.unk_token) for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "tokenizer": {
                "kind": "char",
                "unk_token": self.unk_token,
                "vocab_size": self.vocab_size,
            },
        }


class SentencePieceTokenizer(BaseTokenizer):
    kind: str = "sentencepiece"

    def __init__(self, *, model_bytes: bytes, model_type: str = "bpe"):
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise TokenizerError(
                "sentencepiece tokenizer requested but package is not installed. "
                "Install sentencepiece or use --tokenizer char."
            ) from exc

        self._spm = spm
        self._sp = spm.SentencePieceProcessor()
        loaded = False
        if hasattr(self._sp, "LoadFromSerializedProto"):
            loaded = bool(self._sp.LoadFromSerializedProto(model_bytes))
        if not loaded:
            with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as f:
                f.write(model_bytes)
                tmp_path = f.name
            try:
                loaded = bool(self._sp.Load(tmp_path))
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass
        if not loaded:
            raise TokenizerError("Failed to load sentencepiece model from bytes.")

        self.model_bytes = model_bytes
        self.model_type = model_type

    @staticmethod
    def train_from_corpus(
        *,
        corpus_path: str,
        vocab_size: int,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ) -> "SentencePieceTokenizer":
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise TokenizerError(
                "sentencepiece tokenizer requested but package is not installed. "
                "Install sentencepiece or use --tokenizer char."
            ) from exc

        tmp_root = Path("data/.tmp")
        tmp_root.mkdir(parents=True, exist_ok=True)
        run_dir = (tmp_root / f"spm_{uuid.uuid4().hex[:10]}").resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            model_prefix = str(run_dir / "ickle_spm")
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=max(64, int(vocab_size)),
                model_type=model_type,
                character_coverage=float(character_coverage),
                bos_id=-1,
                eos_id=-1,
            )
            model_file = Path(model_prefix + ".model")
            model_bytes = model_file.read_bytes()
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        return SentencePieceTokenizer(model_bytes=model_bytes, model_type=model_type)

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._sp.EncodeAsIds(text)]

    def decode(self, token_ids: list[int]) -> str:
        return str(self._sp.DecodeIds([int(i) for i in token_ids]))

    @property
    def vocab_size(self) -> int:
        return int(self._sp.GetPieceSize())

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "tokenizer": {
                "kind": "sentencepiece",
                "model_type": self.model_type,
                "vocab_size": self.vocab_size,
            },
            "tokenizer_model_b64": base64.b64encode(self.model_bytes).decode("ascii"),
        }


def tokenizer_from_checkpoint(ckpt: dict[str, Any]) -> BaseTokenizer:
    meta = ckpt.get("tokenizer")
    if isinstance(meta, dict):
        kind = str(meta.get("kind", "char")).strip().lower()
        if kind == "sentencepiece":
            raw_b64 = ckpt.get("tokenizer_model_b64", "")
            if not raw_b64:
                raise TokenizerError("Checkpoint declares sentencepiece tokenizer but tokenizer_model_b64 is missing.")
            model_bytes = base64.b64decode(str(raw_b64).encode("ascii"))
            return SentencePieceTokenizer(
                model_bytes=model_bytes,
                model_type=str(meta.get("model_type", "bpe")),
            )
        if kind == "char":
            stoi = ckpt.get("stoi")
            itos = ckpt.get("itos")
            if not isinstance(stoi, dict) or not isinstance(itos, dict):
                raise TokenizerError("Checkpoint char tokenizer is missing stoi/itos.")
            return CharTokenizer.from_checkpoint(
                stoi=stoi,
                itos=itos,
                unk_token=str(meta.get("unk_token", "?")),
            )

    # Legacy fallback: old checkpoints only had stoi/itos.
    stoi = ckpt.get("stoi")
    itos = ckpt.get("itos")
    if isinstance(stoi, dict) and isinstance(itos, dict):
        return CharTokenizer.from_checkpoint(stoi=stoi, itos=itos)
    raise TokenizerError("Unable to load tokenizer from checkpoint.")


def sanitize_text_for_tokenizer(text: str, tokenizer: BaseTokenizer) -> str:
    if isinstance(tokenizer, CharTokenizer):
        return "".join(ch for ch in text if ch in tokenizer.stoi)
    return text
