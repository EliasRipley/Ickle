"""Data pipeline quality filters: MinHash dedup, language filtering, length/entropy, source-weighted sampling."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", str(text or "").lower())


# Shared between build_clean_corpus.py and open_dataset_ingest.py, which
# used to each carry their own byte-for-byte-identical copy of these sets
# and the low-quality-pair tail checks below -- a fix to one silently never
# reached the other.
ENGLISH_HINT_TOKENS = {
    "the", "and", "to", "of", "in", "for", "is", "that", "you", "with",
    "on", "as", "it", "are", "be", "or", "this", "can", "your", "will",
}

CONTENT_STOPWORDS = ENGLISH_HINT_TOKENS.union(
    {
        "what", "when", "where", "which", "who", "why", "into", "from",
        "have", "about", "there", "than", "they", "them", "their",
    }
)

_BEGININPUT_RE = re.compile(r"\b(?:BEGININPUT|ENDINPUT|SYSTEM PROMPT)\b", re.IGNORECASE)
_REPEATED_WORD_RUN_RE = re.compile(r"\b(\w+)(?:\s+\1){2,}\b")


def has_repeated_word_run(lower_text: str) -> bool:
    return bool(_REPEATED_WORD_RUN_RE.search(lower_text))


def dialogue_pair_has_structural_noise(prompt: str, response: str) -> bool:
    """BEGININPUT/ENDINPUT markers or code fences -- dataset-artifact noise
    that makes a training pair unusable regardless of anything else about it."""
    return bool(
        _BEGININPUT_RE.search(prompt)
        or _BEGININPUT_RE.search(response)
        or "```" in prompt
        or "```" in response
    )


def dialogue_pair_fails_content_checks(prompt: str, response: str) -> bool:
    """The shared tail of both corpus builders' low-quality-pair heuristic:
    non-ASCII ratio, English-hint token density, prompt/response content-word
    overlap, immediate word-repetition runs, and response length/diversity.
    Returns True if the pair should be rejected."""
    p, r = prompt, response
    ascii_ratio_p = sum(1 for ch in p if ord(ch) < 128) / max(1, len(p))
    ascii_ratio_r = sum(1 for ch in r if ord(ch) < 128) / max(1, len(r))
    if ascii_ratio_p < 0.96 or ascii_ratio_r < 0.96:
        return True

    tokens = re.findall(r"[a-zA-Z']+", f"{p.lower()} {r.lower()}")
    if len(tokens) >= 10:
        hits = sum(1 for tok in tokens if tok in ENGLISH_HINT_TOKENS)
        if hits < 3:
            return True

    prompt_tokens = {t for t in re.findall(r"[a-zA-Z']+", p.lower()) if len(t) >= 4 and t not in CONTENT_STOPWORDS}
    if len(prompt_tokens) >= 3:
        response_tokens = {t for t in re.findall(r"[a-zA-Z']+", r.lower()) if len(t) >= 4 and t not in CONTENT_STOPWORDS}
        if not prompt_tokens.intersection(response_tokens):
            return True

    if has_repeated_word_run(r.lower()):
        return True

    words = re.findall(r"[a-zA-Z']+", r.lower())
    if len(words) > 70:
        return True
    if len(words) >= 12:
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio < 0.35:
            return True

    return False


def _shingle(text: str, n: int = 3) -> set[int]:
    tokens = _tokenize_words(text)
    if len(tokens) < n:
        return set()
    hashes: set[int] = set()
    h = hashlib.sha256()
    for i in range(len(tokens) - n + 1):
        h.update(" ".join(tokens[i : i + n]).encode("utf-8"))
        digest = h.digest()[:8]
        hashes.add(struct.unpack("<Q", digest)[0])
        h = hashlib.sha256()
    return hashes


_MINHASH_PERMUTATIONS = 128
_MINHASH_LARGE_PRIME = 2**61 - 1


def _minhash_signature(shingle_set: set[int], seed: int = 42) -> list[int]:
    """Compute a MinHash signature for a set of shingles."""
    if not shingle_set:
        return []

    rng_state = seed
    sig: list[int] = []
    for _ in range(_MINHASH_PERMUTATIONS):
        rng_state = (rng_state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        a = rng_state
        rng_state = (rng_state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        b = rng_state
        min_hash = _MINHASH_LARGE_PRIME
        for x in shingle_set:
            h = (a * x + b) % _MINHASH_LARGE_PRIME
            if h < min_hash:
                min_hash = h
        sig.append(min_hash)
    return sig


def minhash_jaccard(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
    """Return True if two texts are likely near-duplicates (estimated Jaccard >= threshold)."""
    shingles_a = _shingle(text_a, n=3)
    shingles_b = _shingle(text_b, n=3)
    if not shingles_a or not shingles_b:
        return False

    sig_a = _minhash_signature(shingles_a, seed=1337)
    sig_b = _minhash_signature(shingles_b, seed=1337)

    matches = sum(1 for i in range(min(len(sig_a), len(sig_b))) if sig_a[i] == sig_b[i])
    estimated_jaccard = matches / max(1, min(len(sig_a), len(sig_b)))
    return estimated_jaccard >= threshold


def minhash_deduplicate(
    pairs: list[tuple[str, str]],
    threshold: float = 0.85,
) -> list[tuple[str, str]]:
    """Remove near-duplicate (user, assistant) pairs using MinHash."""
    if not pairs:
        return []
    keep: list[tuple[str, str]] = []
    seen_sigs: list[list[int]] = []

    for user, assistant in pairs:
        text = user + " " + assistant
        sig = _minhash_signature(_shingle(text, n=3))
        is_dup = False
        for prev_sig in seen_sigs:
            if not sig or not prev_sig:
                continue
            matches = sum(1 for i in range(min(len(sig), len(prev_sig))) if sig[i] == prev_sig[i])
            est = matches / max(1, min(len(sig), len(prev_sig)))
            if est >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append((user, assistant))
            seen_sigs.append(sig)

    return keep


ENGLISH_CHAR_FREQ = {
    "e": 0.1249, "t": 0.0928, "a": 0.0804, "o": 0.0764,
    "i": 0.0757, "n": 0.0723, "s": 0.0651, "r": 0.0628,
    "h": 0.0505, "l": 0.0407, "d": 0.0382, "c": 0.0334,
    "u": 0.0273, "m": 0.0251, "f": 0.0240, "p": 0.0214,
    "g": 0.0187, "w": 0.0168, "y": 0.0166, "b": 0.0148,
    "v": 0.0105, "k": 0.0054, "x": 0.0023, "j": 0.0016,
    "q": 0.0012, "z": 0.0009,
}


def english_likeness(text: str) -> float:
    """Score 0.0-1.0 for how English-like the text is based on character frequency."""
    lower = str(text or "").lower()
    letters = [ch for ch in lower if ch in ENGLISH_CHAR_FREQ]
    if len(letters) < 10:
        return 0.0

    char_counts = Counter(letters)
    total = len(letters)
    score = 0.0
    max_possible = 0.0
    for char, freq in ENGLISH_CHAR_FREQ.items():
        observed = char_counts.get(char, 0) / max(1, total)
        score += freq * (1.0 - abs(observed - freq))
        max_possible += freq
    return score / max_possible


def is_likely_english(text: str, min_score: float = 0.55) -> bool:
    return english_likeness(text) >= min_score


def shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy."""
    if not text:
        return 0.0
    counts = Counter(str(text))
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def token_entropy(text: str) -> float:
    """Word-level Shannon entropy (higher = richer vocabulary)."""
    tokens = _tokenize_words(text)
    if len(tokens) < 3:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def length_filter(user: str, assistant: str, min_user: int = 4, max_user: int = 300,
                  min_asst: int = 8, max_asst: int = 600) -> bool:
    """Return True if the pair passes length constraints."""
    u_len = len(str(user or "").strip())
    a_len = len(str(assistant or "").strip())
    return min_user <= u_len <= max_user and min_asst <= a_len <= max_asst


def entropy_filter(user: str, assistant: str, min_char_entropy: float = 2.5,
                   max_char_entropy: float = 6.5) -> bool:
    """Return True if character entropy is in a healthy range."""
    text = str(user or "") + " " + str(assistant or "")
    ent = shannon_entropy(text)
    return min_char_entropy <= ent <= max_char_entropy


def language_filter(user: str, assistant: str, min_english_score: float = 0.55) -> bool:
    """Return True if both user and assistant text are likely English."""
    return is_likely_english(user, min_english_score) and is_likely_english(assistant, min_english_score)


def source_weighted_sample(
    pairs_by_source: dict[str, list[tuple[str, str]]],
    n: int,
    *,
    quality_weights: dict[str, float] | None = None,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Sample n pairs across sources using configurable per-source weights.

    Higher weight = more likely to be sampled from that source.
    Defaults to equal weighting if quality_weights is None.
    """
    import random
    rng = random.Random(seed)

    if not pairs_by_source:
        return []

    sources = list(pairs_by_source.keys())
    if quality_weights is None:
        quality_weights = {s: 1.0 for s in sources}

    total_weight = sum(max(0.0, quality_weights.get(s, 1.0)) for s in sources)
    if total_weight <= 0:
        return []

    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = n * 20

    while len(selected) < n and attempts < max_attempts:
        attempts += 1
        r = rng.random() * total_weight
        cumulative = 0.0
        chosen_source = sources[-1]
        for s in sources:
            cumulative += max(0.0, quality_weights.get(s, 1.0))
            if r <= cumulative:
                chosen_source = s
                break

        pool = pairs_by_source.get(chosen_source, [])
        if not pool:
            continue

        pair = rng.choice(pool)
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            selected.append(pair)

    return selected


CORE_CONTAMINATION_PHRASES: tuple[str, ...] = (
    "elias ripley",
    "roman empire",
    "my creator is",
    "from wikipedia, the free encyclopedia",
    "wikipedia does not have an article",
    "article wizard",
    "sister projects",
    "begininput",
    "endinput",
    "system prompt",
)
"""Canonical list of known-bad phrases (author leakage, wiki-scrape boilerplate,
prompt-injection markers) that shouldn't end up in training corpora. Several
corpus builders (sanitize_training_data.py, build_clean_corpus.py,
build_honest_context_package.py) each grew their own copy of this list and
drifted apart in minor ways (word-boundary style, plural handling, extra
file-specific phrases). This is the source of truth for the shared core set;
callers that need file-specific extras or different matching semantics build
their own regex around it rather than being forced into one shape."""


def is_contaminated(text: str) -> bool:
    """Case-insensitive whole-phrase match against CORE_CONTAMINATION_PHRASES."""
    lowered = str(text or "").lower()
    return any(re.search(rf"\b{phrase}\b", lowered, flags=re.IGNORECASE) for phrase in CORE_CONTAMINATION_PHRASES)


def build_quality_filtered_corpus(
    pairs: list[tuple[str, str]],
    *,
    minhash_threshold: float = 0.85,
    min_user_len: int = 4,
    max_user_len: int = 300,
    min_asst_len: int = 8,
    max_asst_len: int = 600,
    min_char_entropy: float = 2.5,
    max_char_entropy: float = 6.5,
    min_english_score: float = 0.55,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Run the full quality pipeline on a list of (user, assistant) pairs.

    Returns filtered pairs and a stats dict.
    """
    stats: dict[str, int] = {"input": len(pairs)}

    filtered = [
        p for p in pairs
        if length_filter(p[0], p[1], min_user_len, max_user_len, min_asst_len, max_asst_len)
    ]
    stats["after_length"] = len(filtered)

    filtered = [
        p for p in filtered
        if entropy_filter(p[0], p[1], min_char_entropy, max_char_entropy)
    ]
    stats["after_entropy"] = len(filtered)

    filtered = [
        p for p in filtered
        if language_filter(p[0], p[1], min_english_score)
    ]
    stats["after_language"] = len(filtered)

    filtered = minhash_deduplicate(filtered, threshold=minhash_threshold)
    stats["after_minhash"] = len(filtered)

    return filtered, stats
