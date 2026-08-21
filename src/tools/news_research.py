from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


@dataclass
class NewsItem:
    title: str
    link: str
    published: str


def search_news(query: str, max_results: int = 5, timeout_sec: int = 12, *, hl: str = "", gl: str = "", ceid: str = "") -> list[NewsItem]:
    """Search Google News RSS for a topic and return top headlines.

    This is a lightweight internet research tool for local agents doing market scans.
    """
    safe_q = quote_plus(query)
    _hl = hl or os.environ.get("ICKLE_NEWS_HL", "en-US")
    _gl = gl or os.environ.get("ICKLE_NEWS_GL", "US")
    _ceid = ceid or os.environ.get("ICKLE_NEWS_CEID", "US:en")
    url = f"https://news.google.com/rss/search?q={safe_q}&hl={_hl}&gl={_gl}&ceid={_ceid}"
    req = Request(url, headers={"User-Agent": "IckleAgent/1.0"})
    with urlopen(req, timeout=timeout_sec) as response:
        xml_bytes = response.read()

    root = ET.fromstring(xml_bytes)
    out: list[NewsItem] = []
    for item in root.findall("./channel/item")[: max(1, max_results)]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        out.append(NewsItem(title=title, link=link, published=pub))
    return out


def format_news_digest(items: list[NewsItem]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        pub = item.published
        try:
            parsed = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
            pub = parsed.strftime("%Y-%m-%d")
        except Exception:
            pass
        lines.append(f"{i}. {item.title} ({pub})\n   {item.link}")
    return "\n".join(lines)
