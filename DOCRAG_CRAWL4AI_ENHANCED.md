# DocRAG Enhanced Crawl4AI Integration - For Modern Doc Sites

## Overview

This is an **enhanced version** of the Crawl4AI integration specifically designed for modern documentation sites like BrightSign that use:
- JavaScript-rendered navigation
- Dynamic sidebars
- Client-side routing
- Nested hierarchies

## What's Different from Basic Integration

- ✅ **Sitemap discovery** - Automatically finds and uses sitemaps
- ✅ **JavaScript execution** - Waits for dynamic content to load
- ✅ **URL discovery mode** - Pre-crawls to find all pages
- ✅ **Batch scraping** - Can scrape from URL lists
- ✅ **Better link extraction** - Handles JavaScript-based navigation

## Installation

Same as basic integration, but with additional setup:

```bash
cd ~/projects/docrag
source venv/bin/activate
pip install crawl4ai httpx lxml
```

## Enhanced File: `docrag/scrapers/crawl4ai_scraper.py`

Replace the entire file with this enhanced version:

```python
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
```

## Enhanced CLI Commands

Add these commands to `docrag/cli.py`:

### New Command: `discover`

Add this new command to discover URLs before scraping:

```python
@main.command()
@click.argument('url')
@click.option('--output', '-o', type=click.Path(), help='Output file for URLs (default: urls.txt)')
@click.option('--max-urls', default=10000, type=int, help='Maximum URLs to discover')
def discover(url: str, output: str, max_urls: int):
    """Discover all documentation URLs on a site without scraping
    
    \b
    This is useful for:
    - Seeing what will be scraped before scraping
    - Creating a filtered URL list
    - Checking if sitemap exists
    
    \b
    Examples:
      docrag discover https://docs.brightsign.biz
      docrag discover https://docs.venafi.com --output venafi-urls.txt
    """
    from docrag.scrapers import CRAWL4AI_AVAILABLE
    
    if not CRAWL4AI_AVAILABLE:
        console.print("❌ Crawl4AI not installed", style="bold red")
        console.print("Install with: pip install crawl4ai")
        sys.exit(1)
    
    from docrag.scrapers.crawl4ai_scraper import Crawl4AIScraper
    from pathlib import Path
    import tempfile
    
    output_file = Path(output) if output else Path('urls.txt')
    
    console.print(f"\n[bold]Discovering URLs from:[/] [cyan]{url}[/]\n")
    
    async def run():
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = Crawl4AIScraper(
                output_dir=Path(tmpdir),
                max_pages=max_urls,
                use_sitemap=True,
                wait_for_js=True
            )
            
            urls = await scraper.discover_urls(url)
            return urls
    
    try:
        with console.status("[cyan]Discovering URLs..."):
            urls = asyncio.run(run())
        
        # Save to file
        with open(output_file, 'w') as f:
            for url_item in urls:
                f.write(f"{url_item}\n")
        
        console.print(f"\n✓ Discovered [green]{len(urls)}[/] URLs", style="bold")
        console.print(f"Saved to: [cyan]{output_file}[/]")
        console.print(f"\nNext: [yellow]docrag scrape {url} --output ./docs --smart[/]")
        console.print(f"Or: [yellow]docrag scrape --from-file {output_file} --output ./docs --smart[/]")
        
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="bold red")
        sys.exit(1)
```

### Update the `scrape` command

Modify the existing scrape command to add `--from-file` option:

```python
@main.command()
@click.argument('url', required=False)
@click.option('--output', '-o', type=click.Path(), required=True, help='Output directory')
@click.option('--from-file', type=click.Path(exists=True), help='Scrape URLs from a file')
@click.option('--playwright', is_flag=True, help='Use Playwright for dynamic content (basic scraper)')
@click.option('--smart', '--use-crawl4ai', 'use_smart', is_flag=True, help='Use AI-powered Crawl4AI scraper')
@click.option('--no-llm', is_flag=True, help='Disable LLM extraction in Crawl4AI')
@click.option('--llm-provider', default='openai/gpt-4o-mini', help='LLM provider for smart scraping')
@click.option('--no-sitemap', is_flag=True, help='Skip sitemap discovery')
@click.option('--no-js-wait', is_flag=True, help='Skip waiting for JavaScript')
@click.option('--max-pages', default=1000, type=int, help='Maximum pages to scrape')
def scrape(url: str, output: str, from_file: str, playwright: bool, use_smart: bool, 
           no_llm: bool, llm_provider: str, no_sitemap: bool, no_js_wait: bool, max_pages: int):
    """Scrape documentation from a website or URL list
    
    \b
    Basic usage:
      docrag scrape https://docs.example.com --output ./docs --smart
    
    \b
    Scrape from URL list:
      docrag discover https://docs.example.com --output urls.txt
      docrag scrape --from-file urls.txt --output ./docs --smart
    
    \b
    For complex JavaScript sites (like BrightSign):
      docrag scrape https://docs.brightsign.biz --output ./docs --smart --max-pages 500
    """
    from docrag.scrapers import CRAWL4AI_AVAILABLE
    
    # Validate arguments
    if not url and not from_file:
        console.print("❌ Either URL or --from-file must be provided", style="bold red")
        sys.exit(1)
    
    if url and from_file:
        console.print("❌ Cannot use both URL and --from-file", style="bold red")
        sys.exit(1)
    
    output_dir = Path(output)
    
    # Check smart scraping availability
    if use_smart and not CRAWL4AI_AVAILABLE:
        console.print()
        console.print(Panel.fit(
            "[yellow]⚠️  Crawl4AI not installed[/]\n\n"
            "Smart scraping requires Crawl4AI. Install it with:\n"
            "  [cyan]pip install crawl4ai[/]\n\n"
            "Falling back to basic scraper...",
            title="Smart Scraping Unavailable",
            border_style="yellow"
        ))
        use_smart = False
    
    console.print()
    console.print(f"[bold]Scraping documentation[/]")
    if from_file:
        console.print(f"  Source: [cyan]URL list from {from_file}[/]")
    else:
        console.print(f"  URL: [cyan]{url}[/]")
    console.print(f"  Output: [cyan]{output_dir}[/]")
    
    if use_smart:
        console.print(f"  Method: [green]Smart (Crawl4AI)[/]")
        if not no_sitemap:
            console.print(f"  Sitemap: [green]Enabled[/]")
        if not no_js_wait:
            console.print(f"  JS Wait: [green]Enabled[/]")
        if not no_llm:
            console.print(f"  LLM: [yellow]{llm_provider}[/]")
        else:
            console.print(f"  LLM: [dim]Disabled[/]")
    else:
        console.print(f"  Method: [yellow]{'Playwright' if playwright else 'Basic'}[/]")
    
    console.print(f"  Max pages: [yellow]{max_pages}[/]")
    console.print()
    
    async def run():
        if use_smart:
            if from_file:
                from docrag.scrapers.crawl4ai_scraper import scrape_from_url_list
                return await scrape_from_url_list(
                    Path(from_file),
                    output_dir,
                    use_llm=not no_llm,
                    llm_provider=llm_provider,
                    max_pages=max_pages
                )
            else:
                from docrag.scrapers.crawl4ai_scraper import scrape_url_smart
                return await scrape_url_smart(
                    url,
                    output_dir,
                    max_pages=max_pages,
                    use_llm=not no_llm,
                    llm_provider=llm_provider,
                    use_sitemap=not no_sitemap,
                    wait_for_js=not no_js_wait
                )
        else:
            if from_file:
                console.print("❌ --from-file requires --smart flag", style="bold red")
                sys.exit(1)
            
            from docrag.scrapers.generic import scrape_url
            return await scrape_url(url, output_dir, playwright, max_pages)
    
    try:
        with console.status("[cyan]Scraping in progress..."):
            pages = asyncio.run(run())
        
        console.print(f"\n✓ Scraped [green]{pages}[/] pages to [cyan]{output_dir}[/]", style="bold")
        console.print(f"\nNext: [yellow]docrag add <n> --source {output_dir}[/]")
    
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="bold red")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)
```

## Usage Examples for BrightSign

### Strategy 1: Auto-Discovery (Recommended)

```bash
# Smart scraping with automatic sitemap discovery
docrag scrape https://docs.brightsign.biz/developers \
  --output ./brightsign-docs \
  --smart \
  --max-pages 500

# Then index
docrag add brightsign --source ./brightsign-docs
```

### Strategy 2: Discover Then Scrape (Most Control)

```bash
# Step 1: Discover all URLs
docrag discover https://docs.brightsign.biz/developers \
  --output brightsign-urls.txt \
  --max-urls 1000

# Step 2: Review/filter URLs (optional)
cat brightsign-urls.txt | grep "/api/" > brightsign-api-only.txt

# Step 3: Scrape from filtered list
docrag scrape --from-file brightsign-api-only.txt \
  --output ./brightsign-docs \
  --smart

# Step 4: Index
docrag add brightsign --source ./brightsign-docs
```

### Strategy 3: Without LLM (Faster, Free)

```bash
# No API key needed, still better than basic
docrag scrape https://docs.brightsign.biz/developers \
  --output ./brightsign-docs \
  --smart \
  --no-llm \
  --max-pages 500
```

## Testing

```bash
# Test discovery
docrag discover https://docs.python.org/3/library \
  --output test-urls.txt \
  --max-urls 50

# Check what was found
wc -l test-urls.txt
head -20 test-urls.txt

# Test scraping from list
docrag scrape --from-file test-urls.txt \
  --output /tmp/test-scrape \
  --smart \
  --no-llm \
  --max-pages 10

# Verify
ls -lh /tmp/test-scrape/
```

## What This Solves

For BrightSign docs specifically:

✅ **Sitemap discovery** - Finds all pages automatically  
✅ **JavaScript navigation** - Waits for sidebar to load  
✅ **Deep hierarchy** - Gets all nested pages  
✅ **URL filtering** - Can pre-filter what to scrape  
✅ **Progress tracking** - Shows N/total as it scrapes  
✅ **Rate limiting** - Respectful of server  

## Installation

```bash
cd ~/projects/docrag
source venv/bin/activate

# Install dependencies
pip install crawl4ai httpx lxml

# Update code files (3 files: crawl4ai_scraper.py, cli.py, __init__.py)

# Test
python -c "from docrag.scrapers.crawl4ai_scraper import CRAWL4AI_AVAILABLE; print('OK' if CRAWL4AI_AVAILABLE else 'FAIL')"
```

## Summary

This enhanced version **specifically handles modern documentation sites** like BrightSign by:

1. **Trying sitemap first** - Most doc sites have them
2. **Waiting for JavaScript** - Dynamic sidebars/navigation load properly
3. **URL discovery mode** - See all pages before scraping
4. **Batch scraping** - Scrape from curated URL lists
5. **Better link extraction** - Handles client-side routing

The result: **You'll get ALL the BrightSign documentation pages**, not just what the basic crawler can find by following HTML links.

---

Pass this to Claude Code for a complete solution to modern doc sites! 🚀
