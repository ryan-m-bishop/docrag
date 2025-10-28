"""Enhanced AI-powered documentation scraper using Crawl4AI"""

from pathlib import Path
from typing import Set, Optional, List
import logging
from urllib.parse import urlparse, urljoin
import asyncio
import httpx

from .base import BaseScraper

logger = logging.getLogger(__name__)

# Check if Crawl4AI is available
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("Crawl4AI not installed. Use 'pip install crawl4ai' for smart scraping.")


class Crawl4AIScraper(BaseScraper):
    """Enhanced AI-powered scraper using Crawl4AI for modern documentation sites"""

    def __init__(
        self,
        output_dir: Path,
        max_pages: int = 1000,
        allowed_domains: Optional[Set[str]] = None,
        use_llm: bool = True,
        llm_provider: str = "openai/gpt-4o-mini",
        rate_limit: float = 1.0,
        use_sitemap: bool = True,
        wait_for_js: bool = True
    ):
        super().__init__(output_dir)

        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "Crawl4AI is not installed. Install it with: pip install crawl4ai"
            )

        self.max_pages = max_pages
        self.allowed_domains = allowed_domains or set()
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.rate_limit = rate_limit
        self.use_sitemap = use_sitemap
        self.wait_for_js = wait_for_js
        self.visited: Set[str] = set()
        self.to_visit: Set[str] = set()

    async def discover_urls(self, start_url: str) -> List[str]:
        """
        Discover all URLs on the site before scraping
        Tries multiple strategies: sitemap, initial page crawl, recursive discovery
        """
        discovered = set()
        parsed = urlparse(start_url)
        base_domain = parsed.netloc

        # Strategy 1: Try sitemap
        if self.use_sitemap:
            sitemap_urls = await self._discover_from_sitemap(start_url)
            if sitemap_urls:
                logger.info(f"Discovered {len(sitemap_urls)} URLs from sitemap")
                discovered.update(sitemap_urls)
                return list(discovered)

        # Strategy 2: Crawl main page and extract all links
        logger.info("No sitemap found, discovering URLs from main page...")
        async with AsyncWebCrawler(
            headless=True,
            verbose=False
        ) as crawler:
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until="networkidle",
                page_timeout=30000,
                delay_before_return_html=2.0 if self.wait_for_js else 0
            )

            result = await crawler.arun(
                url=start_url,
                config=config
            )

            if result.success:
                # Extract all links
                discovered.add(start_url)

                if hasattr(result, 'links') and result.links:
                    for link_data in result.links:
                        href = link_data.get('href', '') if isinstance(link_data, dict) else getattr(link_data, 'href', '')

                        if not href:
                            continue

                        # Make absolute
                        if not href.startswith('http'):
                            href = urljoin(start_url, href)

                        # Filter same domain
                        link_parsed = urlparse(href)
                        if link_parsed.netloc == base_domain:
                            # Remove fragments and query params for deduplication
                            clean_url = f"{link_parsed.scheme}://{link_parsed.netloc}{link_parsed.path}"
                            discovered.add(clean_url)

        logger.info(f"Discovered {len(discovered)} URLs from initial crawl")
        return list(discovered)

    async def _discover_from_sitemap(self, base_url: str) -> List[str]:
        """Try to discover URLs from sitemap.xml"""
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        sitemap_urls = [
            f"{base}/sitemap.xml",
            f"{base}/sitemap_index.xml",
            f"{base}/docs/sitemap.xml",
            f"{base}/api/sitemap.xml",
        ]

        for sitemap_url in sitemap_urls:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(sitemap_url)

                    if response.status_code == 200:
                        from bs4 import BeautifulSoup

                        # Parse XML sitemap
                        soup = BeautifulSoup(response.text, 'xml')

                        # Handle sitemap index (points to other sitemaps)
                        sitemap_locs = soup.find_all('sitemap')
                        if sitemap_locs:
                            all_urls = []
                            for sitemap_loc in sitemap_locs:
                                loc = sitemap_loc.find('loc')
                                if loc:
                                    sub_urls = await self._discover_from_sitemap(loc.text)
                                    all_urls.extend(sub_urls)
                            return all_urls

                        # Handle regular sitemap
                        urls = []
                        for loc in soup.find_all('loc'):
                            url = loc.text.strip()
                            # Filter to only include allowed domains
                            url_parsed = urlparse(url)
                            if not self.allowed_domains or url_parsed.netloc in self.allowed_domains:
                                urls.append(url)

                        if urls:
                            logger.info(f"Found sitemap at {sitemap_url} with {len(urls)} URLs")
                            return urls

            except Exception as e:
                logger.debug(f"Failed to fetch sitemap {sitemap_url}: {e}")
                continue

        return []

    async def scrape(self, start_url: str) -> int:
        """
        Scrape documentation using Crawl4AI with enhanced discovery

        Args:
            start_url: Starting URL to scrape

        Returns:
            Number of pages scraped
        """
        # Set allowed domain
        parsed = urlparse(start_url)
        if not self.allowed_domains:
            self.allowed_domains.add(parsed.netloc)

        # Discover all URLs first
        logger.info("Discovering all documentation URLs...")
        discovered_urls = await self.discover_urls(start_url)

        # Limit to max_pages
        if len(discovered_urls) > self.max_pages:
            logger.warning(f"Found {len(discovered_urls)} URLs, limiting to {self.max_pages}")
            discovered_urls = discovered_urls[:self.max_pages]

        self.to_visit.update(discovered_urls)
        logger.info(f"Will scrape {len(self.to_visit)} pages")

        pages_scraped = 0

        # Create extraction strategy
        if self.use_llm:
            extraction_strategy = LLMExtractionStrategy(
                provider=self.llm_provider,
                api_token=None,  # Uses OPENAI_API_KEY env var
                instruction=(
                    "Extract only the main technical documentation content. "
                    "Ignore navigation menus, sidebars, footers, advertisements, "
                    "cookie notices, and other non-content elements. "
                    "Preserve all headings, code blocks, tables, lists, and technical details. "
                    "Maintain the document structure and hierarchy. "
                    "Include API signatures, parameters, return values, and examples."
                )
            )
        else:
            extraction_strategy = None

        async with AsyncWebCrawler(
            headless=True,
            verbose=False
        ) as crawler:
            while self.to_visit and pages_scraped < self.max_pages:
                url = self.to_visit.pop()

                if url in self.visited:
                    continue

                try:
                    logger.info(f"Scraping [{pages_scraped + 1}/{min(len(discovered_urls), self.max_pages)}]: {url}")

                    # Configure crawler
                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=50,
                        extraction_strategy=extraction_strategy,
                        wait_until="networkidle" if self.wait_for_js else "domcontentloaded",
                        page_timeout=60000,
                        delay_before_return_html=2.0 if self.wait_for_js else 0,
                        remove_overlay_elements=True
                    )

                    # Crawl with Crawl4AI
                    result = await crawler.arun(url=url, config=config)

                    if not result.success:
                        logger.error(f"Failed to crawl {url}: {result.error_message}")
                        self.visited.add(url)
                        continue

                    # Get content
                    if extraction_strategy and result.extracted_content:
                        content = result.extracted_content
                    else:
                        content = result.markdown

                    if not content or len(content.strip()) < 100:
                        logger.warning(f"Insufficient content from {url}, skipping")
                        self.visited.add(url)
                        continue

                    # Save markdown
                    filename = self._url_to_filename(url)
                    self.save_page(filename, content)

                    self.visited.add(url)
                    pages_scraped += 1

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit)

                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    self.visited.add(url)
                    continue

        logger.info(f"Smart scraping complete: {pages_scraped} pages")
        return pages_scraped

    async def scrape_from_list(self, urls: List[str]) -> int:
        """
        Scrape from a pre-defined list of URLs

        Args:
            urls: List of URLs to scrape

        Returns:
            Number of pages scraped
        """
        self.to_visit.update(urls)
        logger.info(f"Scraping {len(urls)} URLs from provided list")

        # Use the main scrape method but skip discovery
        pages_scraped = 0

        # Create extraction strategy
        if self.use_llm:
            extraction_strategy = LLMExtractionStrategy(
                provider=self.llm_provider,
                api_token=None,
                instruction=(
                    "Extract only the main technical documentation content. "
                    "Ignore navigation menus, sidebars, footers, advertisements. "
                    "Preserve all headings, code blocks, tables, lists, and technical details."
                )
            )
        else:
            extraction_strategy = None

        async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
            while self.to_visit and pages_scraped < self.max_pages:
                url = self.to_visit.pop()

                if url in self.visited:
                    continue

                try:
                    logger.info(f"Scraping [{pages_scraped + 1}/{min(len(urls), self.max_pages)}]: {url}")

                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=50,
                        extraction_strategy=extraction_strategy,
                        wait_until="networkidle" if self.wait_for_js else "domcontentloaded",
                        page_timeout=60000,
                        delay_before_return_html=2.0 if self.wait_for_js else 0,
                        remove_overlay_elements=True
                    )

                    result = await crawler.arun(url=url, config=config)

                    if not result.success:
                        logger.error(f"Failed: {url}")
                        self.visited.add(url)
                        continue

                    # Get content
                    if extraction_strategy and result.extracted_content:
                        content = result.extracted_content
                    else:
                        content = result.markdown

                    if content and len(content.strip()) >= 100:
                        filename = self._url_to_filename(url)
                        self.save_page(filename, content)
                        pages_scraped += 1

                    self.visited.add(url)
                    await asyncio.sleep(self.rate_limit)

                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    self.visited.add(url)
                    continue

        return pages_scraped

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to filename, preserving path structure"""
        import re

        parsed = urlparse(url)
        path = parsed.path.strip('/')

        if not path:
            path = 'index'

        # Keep directory structure but replace / with _
        path = path.replace('/', '_')

        # Clean invalid characters
        path = re.sub(r'[<>:"|?*]', '_', path)

        # Limit length
        if len(path) > 200:
            # Keep first and last parts
            path = path[:100] + '_' + path[-99:]

        # Ensure .md extension
        if not path.endswith('.md'):
            path += '.md'

        return path


async def scrape_url_smart(
    url: str,
    output_dir: Path,
    max_pages: int = 1000,
    use_llm: bool = True,
    llm_provider: str = "openai/gpt-4o-mini",
    use_sitemap: bool = True,
    wait_for_js: bool = True
) -> int:
    """
    Helper function to scrape a URL with enhanced Crawl4AI

    Args:
        url: Starting URL
        output_dir: Output directory
        max_pages: Maximum pages to scrape
        use_llm: Whether to use LLM for extraction
        llm_provider: LLM provider string
        use_sitemap: Try to find and use sitemap
        wait_for_js: Wait for JavaScript to load

    Returns:
        Number of pages scraped
    """
    scraper = Crawl4AIScraper(
        output_dir=output_dir,
        max_pages=max_pages,
        use_llm=use_llm,
        llm_provider=llm_provider,
        use_sitemap=use_sitemap,
        wait_for_js=wait_for_js
    )

    return await scraper.scrape(url)


async def scrape_from_url_list(
    url_file: Path,
    output_dir: Path,
    use_llm: bool = True,
    llm_provider: str = "openai/gpt-4o-mini",
    max_pages: int = 10000
) -> int:
    """
    Scrape from a file containing URLs (one per line)

    Args:
        url_file: Path to file with URLs
        output_dir: Output directory
        use_llm: Whether to use LLM for extraction
        llm_provider: LLM provider
        max_pages: Maximum pages to scrape

    Returns:
        Number of pages scraped
    """
    # Read URLs
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    scraper = Crawl4AIScraper(
        output_dir=output_dir,
        max_pages=max_pages,
        use_llm=use_llm,
        llm_provider=llm_provider,
        use_sitemap=False,  # Already have URLs
        wait_for_js=True
    )

    return await scraper.scrape_from_list(urls)
