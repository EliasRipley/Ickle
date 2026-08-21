#!/usr/bin/env python3
"""
Dynamic web reader for Ickle - can handle any webpage structure
"""

import os
import re
from typing import List, Dict
from urllib.request import ProxyHandler, Request, build_opener, urlopen
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from src.evidence_policy import (
    content_tokens,
    domain_reputation,
    evidence_score,
    freshness_score,
    topic_relevance,
)
from src.browser_runtime import launch_headless_browser


class DynamicWebReader:
    """Intelligent web reader that can adapt to any website structure"""
    
    def __init__(self):
        self.content_selectors = [
            # Main content areas
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '.post-content', '.article-content', '.story-content', '.entry-content',
            
            # Common content containers
            '.container', '.wrapper', '.page-content', '.site-content',
            
            # Specific content types
            '.news-article', '.blog-post', '.story', '.post'
        ]
        
        self.headline_selectors = [
            # Headline hierarchy
            'h1', 'h2', 'h3',
            
            # Common headline classes
            '.headline', '.title', '.story-title', '.post-title',
            '.article-title', '.entry-title', '.news-title',
            
            # Specific headline patterns
            '[data-testid="headline"]', '.headline-text',
            '.story-headline', '.post-headline'
        ]
        
        self.navigation_selectors = [
            'nav', '.nav', '.navigation', '.menu', '.navbar',
            '.header-nav', '.top-nav', '.main-menu'
        ]
        
        self.sidebar_selectors = [
            'aside', '.sidebar', '.widget', '.side-content',
            '.secondary', '.complementary'
        ]
        
        self.footer_selectors = [
            'footer', '.footer', '.site-footer', '.page-footer'
        ]
    
    def read_url(self, url: str, timeout_ms: int = 30000, max_chars: int = 15000) -> Dict:
        """Read and analyze any webpage structure"""
        urllib_result = self._read_with_urllib(url=url, timeout_ms=timeout_ms, max_chars=max_chars)
        if urllib_result.get("success"):
            return urllib_result

        if self._looks_like_network_block(urllib_result.get("error", "")):
            return urllib_result

        try:
            with sync_playwright() as p:
                browser = None
                try:
                    browser, _browser_name = launch_headless_browser(p, headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=timeout_ms)
                    page.wait_for_load_state("domcontentloaded")
                    page.wait_for_timeout(2000)
                    html = page.content()
                    result = self._analyze_html(url, html, max_chars=max_chars)
                    result["reader_mode"] = "playwright"
                    return result
                finally:
                    if browser is not None:
                        browser.close()
        except Exception as playwright_error:
            return {
                'url': url,
                'error': (
                    f"HTTP fetch failed: {urllib_result.get('error', 'unknown error')}. "
                    f"Playwright failed: {playwright_error}."
                ),
                'success': False
            }

    def _analyze_html(self, url: str, html: str, max_chars: int = 15000) -> Dict:
        soup = BeautifulSoup(html, 'html.parser')
        analysis = self.analyze_page_structure(soup, url)
        main_content = self.extract_main_content(soup)
        if len(main_content) > max_chars:
            main_content = main_content[:max_chars]
        headlines = self.extract_headlines(soup)
        metadata = self.extract_metadata(soup, url)
        quality = self._source_quality(url=url, metadata=metadata, analysis=analysis, content=main_content)
        evidence_items = self._build_evidence_items(
            url=url,
            title=metadata.get('title', ''),
            metadata=metadata,
            headlines=headlines,
            main_content=main_content,
            source_quality=quality,
        )
        return {
            'url': url,
            'title': metadata.get('title', 'No title'),
            'description': metadata.get('description', ''),
            'structure': analysis,
            'headlines': headlines,
            'content': main_content,
            'word_count': len(main_content.split()),
            'source_quality': quality,
            'evidence_items': evidence_items,
            'success': True
        }

    def _read_with_urllib(self, url: str, timeout_ms: int, max_chars: int) -> Dict:
        """Fast path web reader that does not require browser process launch."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": os.environ.get("ICKLE_ACCEPT_LANGUAGE", "en-US,en;q=0.9"),
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            request = Request(url, headers=headers)
            timeout_sec = max(5, int(timeout_ms / 1000))
            proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or ""
            if proxy_url:
                proxy_handler = ProxyHandler({"https": proxy_url, "http": proxy_url})
                opener = build_opener(proxy_handler)
                with opener.open(request, timeout=timeout_sec) as response:
                    raw = response.read(2_000_000)
                    encoding = response.headers.get_content_charset() or "utf-8"
            else:
                with urlopen(request, timeout=timeout_sec) as response:
                    raw = response.read(2_000_000)
                    encoding = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(encoding, errors="replace")
            result = self._analyze_html(url, html, max_chars=max_chars)
            result["reader_mode"] = "urllib"
            return result
        except Exception as fetch_error:
            return {
                'url': url,
                'error': f"{fetch_error}",
                'success': False
            }

    def _source_quality(self, *, url: str, metadata: Dict, analysis: Dict, content: str) -> Dict:
        domain = (urlparse(url).hostname or "").lower()
        domain_score = domain_reputation(url)
        published = str(metadata.get("published_date", "") or metadata.get("article_published", "")).strip()
        fresh = freshness_score(published)
        structure_score = 0.35
        if analysis.get("has_main"):
            structure_score += 0.20
        if analysis.get("has_header"):
            structure_score += 0.08
        if analysis.get("has_nav"):
            structure_score += 0.08
        if analysis.get("has_footer"):
            structure_score += 0.06
        if len(content.split()) >= 180:
            structure_score += 0.18
        elif len(content.split()) >= 80:
            structure_score += 0.10
        structure_score = max(0.0, min(1.0, structure_score))
        score = (0.45 * domain_score) + (0.30 * fresh) + (0.25 * structure_score)
        return {
            "domain": domain,
            "domain_score": round(domain_score, 4),
            "freshness_score": round(fresh, 4),
            "structure_score": round(structure_score, 4),
            "score": round(max(0.0, min(1.0, score)), 4),
        }

    def _build_evidence_items(
        self,
        *,
        url: str,
        title: str,
        metadata: Dict,
        headlines: List[Dict],
        main_content: str,
        source_quality: Dict,
    ) -> List[Dict]:
        out: List[Dict] = []
        source_score = float(source_quality.get("score", 0.6))
        published = str(metadata.get("published_date", "") or metadata.get("article_published", "")).strip()
        title_tokens = content_tokens(title)

        for row in headlines[:12]:
            claim = str(row.get("text", "")).strip()
            if len(claim) < 24:
                continue
            relevance = 0.35
            if title_tokens:
                relevance = max(relevance, topic_relevance(title, claim))
            score = evidence_score(
                relevance=relevance,
                source_quality=source_score,
                confidence=0.84,
                corroboration_count=0,
            )
            out.append(
                {
                    "claim": claim,
                    "kind": "headline",
                    "source_url": url,
                    "source_title": title,
                    "published_date": published or None,
                    "relevance": round(relevance, 4),
                    "confidence": 0.84,
                    "corroboration_count": 0,
                    "score": round(score, 4),
                }
            )

        candidate_sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", main_content):
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) < 50 or len(cleaned) > 260:
                continue
            if cleaned.lower().startswith(("text:", "item:")):
                cleaned = cleaned.split(":", 1)[1].strip()
            candidate_sentences.append(cleaned)
            if len(candidate_sentences) >= 14:
                break

        for claim in candidate_sentences:
            relevance = 0.30
            if title_tokens:
                relevance = max(relevance, topic_relevance(title, claim))
            score = evidence_score(
                relevance=relevance,
                source_quality=source_score,
                confidence=0.74,
                corroboration_count=0,
            )
            out.append(
                {
                    "claim": claim,
                    "kind": "content",
                    "source_url": url,
                    "source_title": title,
                    "published_date": published or None,
                    "relevance": round(relevance, 4),
                    "confidence": 0.74,
                    "corroboration_count": 0,
                    "score": round(score, 4),
                }
            )

        out.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        deduped: List[Dict] = []
        seen: set[str] = set()
        for item in out:
            key = str(item.get("claim", "")).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= 18:
                break
        return deduped

    def _looks_like_network_block(self, error_text: str) -> bool:
        lowered = str(error_text).lower()
        signals = [
            "10013",
            "forbidden by its access permissions",
            "name or service not known",
            "temporary failure in name resolution",
            "network is unreachable",
            "connection refused",
        ]
        return any(token in lowered for token in signals)
    
    def analyze_page_structure(self, soup: BeautifulSoup, url: str) -> Dict:
        """Analyze the webpage structure"""
        
        structure = {
            'has_main': bool(soup.select_one('main, article, [role="main"]')),
            'has_header': bool(soup.select_one('header')),
            'has_nav': bool(soup.select_one('nav')),
            'has_aside': bool(soup.select_one('aside')),
            'has_footer': bool(soup.select_one('footer')),
            'content_areas': [],
            'headline_elements': [],
            'framework_detected': None
        }
        
        # Detect common frameworks
        if soup.select_one('[data-reactroot], [data-vue], [ng-app]'):
            structure['framework_detected'] = 'SPA (React/Vue/Angular)'
        
        # Find content areas
        for selector in self.content_selectors:
            elements = soup.select(selector)
            if elements:
                structure['content_areas'].extend([elem.name for elem in elements[:3]])
        
        # Find headline elements
        for selector in self.headline_selectors:
            elements = soup.select(selector)
            if elements:
                structure['headline_elements'].extend([elem.name for elem in elements[:3]])
        
        return structure
    
    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from any webpage"""
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'noscript', 'iframe', 'svg']):
            element.decompose()
        
        # Remove navigation, sidebar, footer
        for selector in self.navigation_selectors + self.sidebar_selectors + self.footer_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Try to find main content area
        main_content = None
        
        # Priority order for content extraction
        content_selectors = [
            'main article',
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.article-content',
            '.story-content',
            '.entry-content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        # If no specific content area found, use body but clean it
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text with structure preservation
        content_parts = []
        
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'a', 'li']):
            text = element.get_text(strip=True)
            if text and len(text) > 5:  # Skip very short text
                # Add structural markers
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    content_parts.append(f"HEADLINE: {text}")
                elif element.name == 'p':
                    content_parts.append(f"PARAGRAPH: {text}")
                elif element.name == 'li':
                    content_parts.append(f"ITEM: {text}")
                else:
                    content_parts.append(f"TEXT: {text}")
        
        return ' '.join(content_parts)
    
    def extract_headlines(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract headlines from any webpage structure"""
        
        headlines = []
        
        # Remove unwanted elements first
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Look for headlines in priority order
        headline_selectors = [
            'h1', 'h2', 'h3',
            '.headline', '.title', '.story-title', '.post-title',
            '.article-title', '.entry-title', '.news-title'
        ]
        
        seen_headlines = set()
        
        for selector in headline_selectors:
            elements = soup.select(selector)
            
            for element in elements[:10]:  # Limit per selector
                text = element.get_text(strip=True)
                
                # Skip if too short, too long, or already seen
                if (len(text) < 10 or len(text) > 200 or 
                    text.lower() in seen_headlines):
                    continue
                
                # Skip common non-headline text
                skip_patterns = [
                    r'^\s*(skip|menu|home|search|contact|about|login|register)\s*$',
                    r'^\s*\d+\s*$',
                    r'^\s*[^\w\s]+\s*$'  # Just symbols
                ]
                
                if any(re.match(pattern, text, re.IGNORECASE) for pattern in skip_patterns):
                    continue
                
                # Determine headline level
                if element.name == 'h1':
                    level = 1
                elif element.name == 'h2':
                    level = 2
                elif element.name == 'h3':
                    level = 3
                else:
                    level = 4  # Other headline-like elements
                
                headlines.append({
                    'text': text,
                    'level': level,
                    'selector': selector,
                    'element': element.name
                })
                
                seen_headlines.add(text.lower())
        
        # Sort by level and length (longer headlines often more important)
        headlines.sort(key=lambda x: (x['level'], -len(x['text'])))
        
        return headlines[:15]  # Return top 15 headlines
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from webpage"""
        
        metadata = {}
        
        # Title
        title_elem = soup.find('title')
        metadata['title'] = title_elem.get_text(strip=True) if title_elem else ''
        
        # Meta description
        desc_elem = soup.find('meta', attrs={'name': 'description'})
        metadata['description'] = desc_elem.get('content', '') if desc_elem else ''
        
        # Meta keywords
        keywords_elem = soup.find('meta', attrs={'name': 'keywords'})
        metadata['keywords'] = keywords_elem.get('content', '') if keywords_elem else ''
        
        # Open Graph data
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        og_description = soup.find('meta', property='og:description')
        if og_description:
            metadata['og_description'] = og_description.get('content', '')
        
        # Author
        author_elem = soup.find('meta', attrs={'name': 'author'})
        if author_elem:
            metadata['author'] = author_elem.get('content', '')
        
        # Published date. A single attrs={} dict on soup.find() ANDs the keys
        # together -- requiring both name="date" AND property=
        # "article:published_time" on the very same <meta> tag, which real
        # pages never do (they set one or the other). Try each real-world
        # variant separately so any one of them matches.
        pub_date_elem = (
            soup.find('meta', attrs={'property': 'article:published_time'})
            or soup.find('meta', attrs={'name': 'date'})
            or soup.find('meta', attrs={'name': 'pubdate'})
            or soup.find('meta', attrs={'name': 'publish-date'})
            or soup.find('meta', attrs={'property': 'og:published_time'})
        )
        if pub_date_elem:
            metadata['published_date'] = pub_date_elem.get('content', '')
        
        return metadata


def read_url_dynamic(url: str, timeout_ms: int = 30000, max_chars: int = 15000) -> Dict:
    """Convenience function for dynamic web reading"""
    reader = DynamicWebReader()
    return reader.read_url(url, timeout_ms, max_chars)
