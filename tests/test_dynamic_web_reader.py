import unittest

from bs4 import BeautifulSoup

from src.dynamic_web_reader import DynamicWebReader


class DynamicWebReaderMetadataTests(unittest.TestCase):
    """Regression coverage for a real bug: extract_metadata() searched for
    the published date with soup.find('meta', attrs={'name': 'date',
    'property': 'article:published_time'}) -- a single attrs dict ANDs its
    keys, requiring both attributes on the very same <meta> tag. Real pages
    set only one or the other, so this never matched anything in practice."""

    def test_finds_published_date_from_article_published_time_property(self):
        html = '<html><head><meta property="article:published_time" content="2024-01-15"></head></html>'
        soup = BeautifulSoup(html, "html.parser")
        metadata = DynamicWebReader().extract_metadata(soup, "https://example.com")
        self.assertEqual(metadata.get("published_date"), "2024-01-15")

    def test_finds_published_date_from_name_date(self):
        html = '<html><head><meta name="date" content="2024-02-20"></head></html>'
        soup = BeautifulSoup(html, "html.parser")
        metadata = DynamicWebReader().extract_metadata(soup, "https://example.com")
        self.assertEqual(metadata.get("published_date"), "2024-02-20")


class DynamicWebReaderTests(unittest.TestCase):
    def test_analyze_html_extracts_basic_fields(self):
        reader = DynamicWebReader()
        html = """
        <html>
          <head>
            <title>Test Page</title>
            <meta name="description" content="A short page for testing.">
          </head>
          <body>
            <header>Header</header>
            <main>
              <h1>Main Heading</h1>
              <p>First paragraph of useful content.</p>
            </main>
          </body>
        </html>
        """
        result = reader._analyze_html("https://example.com", html, max_chars=500)
        self.assertTrue(result["success"])
        self.assertEqual(result["title"], "Test Page")
        self.assertIn("Main Heading", " ".join(h["text"] for h in result["headlines"]))
        self.assertGreater(result["word_count"], 0)


if __name__ == "__main__":
    unittest.main()
