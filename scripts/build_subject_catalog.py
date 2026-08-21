from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

SOURCE_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=parse&page=Outline_of_academic_disciplines&prop=wikitext&formatversion=2&format=json"
)

IGNORE_TOP_SECTIONS = {"see also", "notes", "further reading", "external links"}


def _fetch_wikitext(url: str = SOURCE_URL) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "IckleSubjectCatalogBuilder/1.0 (+https://github.com/EliasRipley/Ickle)",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=40) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    return str(payload["parse"]["wikitext"])


def _clean_wikitext_fragment(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<ref[^/>]*/\s*>", " ", t, flags=re.I)
    for _ in range(6):
        nt = re.sub(r"\{\{[^{}]*\}\}", " ", t)
        if nt == t:
            break
        t = nt

    def _wiki_link(match: re.Match[str]) -> str:
        inner = str(match.group(1) or "").strip()
        if not inner:
            return " "
        target = inner.split("|", 1)[0].strip().lower()
        if any(target.startswith(prefix) for prefix in ("file:", "image:", "category:", "template:", "help:")):
            return " "
        shown = inner.split("|")[-1].strip()
        shown = shown.split("#")[0].strip()
        return shown or " "

    t = re.sub(r"\[\[([^\]]+)\]\]", _wiki_link, t)
    t = re.sub(r"\[(https?://\S+)\s+([^\]]+)\]", r"\2", t)
    t = re.sub(r"\[(https?://[^\]]+)\]", " ", t)
    t = t.replace("&nbsp;", " ").replace("''", "")
    t = re.sub(r"\((?:outline|academic discipline)\)", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -:;,.")
    return t


def _heading_text(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", line)
    if not match:
        return None
    level = len(match.group(1))
    title = _clean_wikitext_fragment(match.group(2))
    if not title:
        return None
    return level, title


def _normalize_item_text(text: str) -> str:
    out = _clean_wikitext_fragment(text)
    out = re.split(r"\s+-\s+|\s+—\s+|\s+–\s+", out, maxsplit=1)[0].strip()
    out = re.sub(r"^(?:main article|main articles)\s*:\s*", "", out, flags=re.I).strip()
    out = re.sub(r"\s+\[\s*\d+\s*\]\s*$", "", out).strip()
    out = re.sub(r"\s+", " ", out).strip(" -:;,.")
    return out


def _build_catalog(wikitext: str, *, max_topics_per_subject: int = 180) -> dict[str, Any]:
    # domain -> subject -> topics
    matrix: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    current = {2: "", 3: "", 4: "", 5: ""}

    def _active_subjects() -> list[str]:
        out: list[str] = []
        for level in (3, 4, 5):
            value = str(current.get(level, "") or "").strip()
            if value and value not in out:
                out.append(value)
        return out

    for raw_line in wikitext.splitlines():
        line = str(raw_line or "")
        head = _heading_text(line)
        if head:
            level, title = head
            if level == 2:
                current[2] = title
                current[3] = current[4] = current[5] = ""
            elif level == 3:
                current[3] = title
                current[4] = current[5] = ""
                domain = current[2]
                if domain and domain.lower() not in IGNORE_TOP_SECTIONS:
                    matrix[domain][title]
            elif level == 4:
                current[4] = title
                current[5] = ""
                domain = current[2]
                if domain and domain.lower() not in IGNORE_TOP_SECTIONS:
                    matrix[domain][title]
                    if current[3]:
                        matrix[domain][current[3]].add(title)
            elif level == 5:
                current[5] = title
                domain = current[2]
                if domain and domain.lower() not in IGNORE_TOP_SECTIONS:
                    matrix[domain][title]
                    if current[4]:
                        matrix[domain][current[4]].add(title)
                    if current[3]:
                        matrix[domain][current[3]].add(title)
            continue

        if not line.startswith("*"):
            continue
        domain = str(current.get(2, "") or "").strip()
        if not domain or domain.lower() in IGNORE_TOP_SECTIONS:
            continue

        depth = len(line) - len(line.lstrip("*"))
        body = _normalize_item_text(line[depth:])
        if not body:
            continue
        if body.lower() in {"edit", "field ministry"}:
            continue
        active = _active_subjects()
        if not active:
            continue
        for subject in active:
            matrix[domain][subject].add(body)

    domains_out: list[dict[str, Any]] = []
    flat_subjects: list[dict[str, Any]] = []
    for domain_name in sorted(matrix.keys()):
        subject_rows: list[dict[str, Any]] = []
        for subject_name in sorted(matrix[domain_name].keys()):
            topics = sorted(
                {
                    re.sub(r"\s+", " ", t).strip()
                    for t in matrix[domain_name][subject_name]
                    if re.sub(r"\s+", " ", str(t)).strip()
                }
            )
            if not topics:
                continue
            topics = topics[: max(8, int(max_topics_per_subject))]
            subject_row = {
                "name": subject_name,
                "topics": topics,
            }
            subject_rows.append(subject_row)
            flat_subjects.append(
                {
                    "name": f"{domain_name}: {subject_name}",
                    "domain": domain_name,
                    "topics": topics,
                }
            )
        if not subject_rows:
            continue
        domains_out.append(
            {
                "name": domain_name,
                "subject_count": len(subject_rows),
                "subjects": subject_rows,
            }
        )

    return {
        "version": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": SOURCE_URL,
            "article": "https://en.wikipedia.org/wiki/Outline_of_academic_disciplines",
            "method": "wikipedia_parse_wikitext",
        },
        "domain_count": len(domains_out),
        "subject_count": len(flat_subjects),
        "domains": domains_out,
        # Backward-compatible flat projection used by runtime selection.
        "subjects": flat_subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build subject catalog from Wikipedia discipline outline.")
    parser.add_argument("--out", default="IckleTraining/subject_catalog.json")
    parser.add_argument("--max-topics-per-subject", type=int, default=180)
    args = parser.parse_args()

    wikitext = _fetch_wikitext()
    catalog = _build_catalog(wikitext, max_topics_per_subject=max(30, int(args.max_topics_per_subject)))
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {out_path} "
        f"(domains={catalog.get('domain_count', 0)}, subjects={catalog.get('subject_count', 0)})"
    )


if __name__ == "__main__":
    main()
