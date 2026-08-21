from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from src.browser_runtime import launch_headless_browser


def read_url_text(url: str, timeout_ms: int = 15_000, max_chars: int = 5_000) -> str:
    """Open a URL with Firefox and extract readable text."""
    with sync_playwright() as p:
        browser, _browser_name = launch_headless_browser(p, headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=timeout_ms)
            page.wait_for_load_state("domcontentloaded")
            html = page.content()
        finally:
            browser.close()

    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]
