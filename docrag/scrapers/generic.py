from .base import BaseScraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import asyncio
from typing import Set, Optional
import logging
import re

logger = logging.getLogger(__name__)


class GenericDocScraper(BaseScraper):
    """Generic scraper for documentation sites"""

    def __init__(
        self,
        output_dir: Path,
        max_pages: int = 1000,
        allowed_domains: Optional[Set[str]] = None,
        use_playwright: bool = False
    ):
        super().__init__(output_dir)
        self.max_pages = max_pages
        self.allowed_domains = allowed_domains or set()
        self.use_playwright = use_playwright
        self.visited: Set[str] = set()
        self.to_visit: Set[str] = set()

    async def scrape(self, start_url: str) -> int:
        """Scrape documentation starting from start_url"""
        # Set allowed domain from start URL if not specified
        if not self.allowed_domains:
            parsed = urlparse(start_url)
            self.allowed_domains.add(parsed.netloc)

        self.to_visit.add(start_url)
        pages_scraped = 0

        while self.to_visit and pages_scraped < self.max_pages:
            url = self.to_visit.pop()

            if url in self.visited:
                continue

            try:
                logger.info(f"Scraping: {url}")

                # Fetch content
                if self.use_playwright:
                    html = await self.fetch_with_playwright(url)
                else:
                    html = await self.fetch_static(url)

                # Parse and extract
                soup = BeautifulSoup(html, 'lxml')

                # Extract main content (try common selectors)
                main_content = self._extract_main_content(soup)

                if main_content:
                    # Convert to markdown
                    markdown = self.html_to_markdown(str(main_content))

                    # Generate filename
                    filename = self._url_to_filename(url)

                    # Save
                    self.save_page(filename, markdown)
                    pages_scraped += 1

                # Find links
                links = self._extract_links(soup, url)
                self.to_visit.update(links)

                self.visited.add(url)

                # Be polite
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                self.visited.add(url)  # Mark as visited to avoid retrying

        return pages_scraped

    def _extract_main_content(self, soup: BeautifulSoup):
        """Extract main documentation content from page"""
        # Try common selectors for documentation sites
        selectors = [
            'main',
            'article',
            '[role="main"]',
            '.main-content',
            '.content',
            '.documentation',
            '#content',
            '#main-content',
        ]

        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                return content

        # Fallback to body
        return soup.body

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract and filter links from page"""
        links = set()

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']

            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)

            # Parse URL
            parsed = urlparse(absolute_url)

            # Filter out anchors, external links, and non-http(s)
            if (
                parsed.scheme in ('http', 'https') and
                parsed.netloc in self.allowed_domains and
                not parsed.fragment and
                absolute_url not in self.visited
            ):
                # Remove query parameters for cleaner URLs
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.add(clean_url)

        return links

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')

        if not path:
            path = 'index'

        # Replace slashes with underscores
        filename = path.replace('/', '_')

        # Remove invalid characters
        filename = re.sub(r'[^\w\-_.]', '_', filename)

        # Ensure .md extension
        if not filename.endswith('.md'):
            filename += '.md'

        return filename


async def scrape_url(
    url: str,
    output_dir: Path,
    use_playwright: bool = False,
    max_pages: int = 1000
) -> int:
    """
    Helper function to scrape a URL with GenericDocScraper

    Args:
        url: Starting URL
        output_dir: Output directory
        use_playwright: Whether to use Playwright
        max_pages: Maximum pages to scrape

    Returns:
        Number of pages scraped
    """
    scraper = GenericDocScraper(
        output_dir=output_dir,
        max_pages=max_pages,
        use_playwright=use_playwright
    )

    return await scraper.scrape(url)
