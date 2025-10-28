"""AI-powered documentation scraper using Crawl4AI"""

from pathlib import Path
from typing import Set, Optional
import logging
from urllib.parse import urlparse, urljoin
import asyncio
import re

from .base import BaseScraper

logger = logging.getLogger(__name__)

# Check if Crawl4AI is available
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("Crawl4AI not installed. Use 'pip install crawl4ai' for smart scraping.")


class Crawl4AIScraper(BaseScraper):
    """AI-powered scraper using Crawl4AI for intelligent content extraction"""
    
    def __init__(
        self,
        output_dir: Path,
        max_pages: int = 1000,
        allowed_domains: Optional[Set[str]] = None,
        use_llm: bool = True,
        llm_provider: str = "openai/gpt-4o-mini",
        rate_limit: float = 1.0
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
        self.visited: Set[str] = set()
        self.to_visit: Set[str] = set()
    
    async def scrape(self, start_url: str) -> int:
        """
        Scrape documentation using Crawl4AI
        
        Args:
            start_url: Starting URL to scrape
            
        Returns:
            Number of pages scraped
        """
        # Set allowed domain
        parsed = urlparse(start_url)
        if not self.allowed_domains:
            self.allowed_domains.add(parsed.netloc)
        
        self.to_visit.add(start_url)
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
                    "Maintain the document structure and hierarchy."
                )
            )
        else:
            # Use simpler extraction without LLM
            extraction_strategy = None
        
        async with AsyncWebCrawler(verbose=False) as crawler:
            while self.to_visit and pages_scraped < self.max_pages:
                url = self.to_visit.pop()
                
                if url in self.visited:
                    continue
                
                try:
                    logger.info(f"Smart scraping [{pages_scraped + 1}/{self.max_pages}]: {url}")
                    
                    # Crawl with Crawl4AI
                    result = await crawler.arun(
                        url=url,
                        word_count_threshold=50,
                        extraction_strategy=extraction_strategy,
                        bypass_cache=True,
                        remove_overlay_elements=True
                    )
                    
                    if not result.success:
                        logger.error(f"Failed to crawl {url}: {result.error_message}")
                        self.visited.add(url)
                        continue
                    
                    # Get content - use extracted content if available, otherwise markdown
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
                    
                    # Extract links for further crawling
                    new_links = self._extract_links_from_result(result, url)
                    self.to_visit.update(new_links)
                    
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
    
    def _extract_links_from_result(self, result, base_url: str) -> Set[str]:
        """Extract valid documentation links from crawl result"""
        links = set()
        
        if not hasattr(result, 'links') or not result.links:
            return links
        
        base_parsed = urlparse(base_url)
        
        for link_data in result.links:
            # Handle both dict and object formats
            if isinstance(link_data, dict):
                href = link_data.get('href', '')
            else:
                href = getattr(link_data, 'href', '')
            
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Parse URL
            parsed = urlparse(href)
            
            # Make absolute if relative
            if not parsed.netloc:
                href = urljoin(base_url, href)
                parsed = urlparse(href)
            
            # Remove fragments
            href = href.split('#')[0]
            
            # Filter
            if (
                parsed.netloc in self.allowed_domains and
                parsed.scheme in ('http', 'https') and
                href not in self.visited and
                not self._is_excluded_path(parsed.path)
            ):
                links.add(href)
        
        return links
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path should be excluded"""
        excluded_extensions = {
            '.pdf', '.zip', '.tar', '.gz', '.exe', '.dmg',
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
            '.mp4', '.mp3', '.avi', '.mov',
            '.css', '.js', '.woff', '.ttf'
        }
        
        path_lower = path.lower()
        
        # Check extensions
        if any(path_lower.endswith(ext) for ext in excluded_extensions):
            return True
        
        # Check for non-doc paths
        excluded_patterns = [
            '/download/', '/downloads/',
            '/login', '/signup', '/register',
            '/search', '/tags/', '/categories/'
        ]
        
        if any(pattern in path_lower for pattern in excluded_patterns):
            return True
        
        return False
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to filename"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if not path:
            path = 'index'
        
        # Replace slashes
        path = path.replace('/', '_')
        
        # Clean invalid characters
        path = re.sub(r'[<>:"|?*]', '_', path)
        
        # Limit length
        if len(path) > 200:
            path = path[:200]
        
        # Ensure .md extension
        if not path.endswith('.md'):
            path += '.md'
        
        return path


async def scrape_url_smart(
    url: str,
    output_dir: Path,
    max_pages: int = 1000,
    use_llm: bool = True,
    llm_provider: str = "openai/gpt-4o-mini"
) -> int:
    """
    Helper function to scrape a URL with Crawl4AI
    
    Args:
        url: Starting URL
        output_dir: Output directory
        max_pages: Maximum pages to scrape
        use_llm: Whether to use LLM for extraction
        llm_provider: LLM provider string (e.g., "openai/gpt-4o-mini")
        
    Returns:
        Number of pages scraped
    """
    scraper = Crawl4AIScraper(
        output_dir=output_dir,
        max_pages=max_pages,
        use_llm=use_llm,
        llm_provider=llm_provider
    )
    
    return await scraper.scrape(url)
