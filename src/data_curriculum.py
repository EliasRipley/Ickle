import re
import math
from typing import Any


def estimate_sample_difficulty(text: str) -> float:
    if not text or len(text) < 10:
        return 0.0
    word_lengths = [len(w) for w in re.findall(r"\w+", text)]
    if not word_lengths:
        return 0.0
    avg_word_len = sum(word_lengths) / len(word_lengths)
    sentences = re.split(r"[.!?]+", text)
    if not sentences:
        return 0.0
    avg_sent_len = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len(sentences))
    rare_chars = sum(1 for c in text if c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\"'-\n")
    rare_ratio = rare_chars / max(1, len(text))
    return avg_word_len * 1.0 + avg_sent_len * 0.5 + rare_ratio * 50.0


def curriculum_sort_samples(samples: list[str], ascending: bool = True) -> list[str]:
    if not samples:
        return []
    scored = [(estimate_sample_difficulty(s), s) for s in samples]
    scored.sort(key=lambda x: x[0], reverse=not ascending)
    return [s for _, s in scored]


def curriculum_sort_file(input_path: str, output_path: str = "", ascending: bool = True,
                          chunk_size: int = 4096):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if len(paragraphs) < 2:
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        return {"samples": 1, "sorted": False}

    sorted_paragraphs = curriculum_sort_samples(paragraphs, ascending=ascending)

    out_text = "\n\n".join(sorted_paragraphs)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(out_text)

    return {"samples": len(paragraphs), "sorted": True, "direction": "easy_first" if ascending else "hard_first"}
