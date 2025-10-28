"""Archbee-specific scraper that uses Playwright to extract navigation"""

from pathlib import Path
from typing import Set, List
import logging
import asyncio

logger = logging.getLogger(__name__)

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Use 'pip install playwright' and 'playwright install chromium'")


async def discover_archbee_urls(start_url: str, max_urls: int = 1000) -> List[str]:
    """
    Discover URLs from Archbee documentation by using Playwright to:
    1. Load the page with JavaScript
    2. Wait for navigation to render
    3. Interact with navigation elements
    4. Extract all documentation links

    Args:
        start_url: Base URL of the Archbee docs
        max_urls: Maximum URLs to discover

    Returns:
        List of discovered URLs
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright is required for Archbee scraping. Install with: pip install playwright && playwright install chromium")

    from urllib.parse import urlparse

    urls = set()
    base_domain = urlparse(start_url).netloc

    logger.info(f"Discovering Archbee URLs from {start_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Load the page and wait for navigation
            logger.info("Loading page and waiting for JavaScript navigation...")
            await page.goto(start_url, wait_until="networkidle", timeout=60000)

            # Wait for navigation elements to load
            await page.wait_for_timeout(3000)

            # Try to expand all navigation items
            # Archbee typically uses buttons or divs with click handlers
            logger.info("Expanding navigation sections...")

            # Strategy 1: Find and click all expandable buttons/elements
            expandable_selectors = [
                'button[aria-expanded="false"]',
                '[class*="expand"]',
                '[class*="toggle"]',
                '[class*="collaps"]',
                'nav button',
                'aside button',
                '[role="button"]'
            ]

            for selector in expandable_selectors:
                elements = await page.query_selector_all(selector)
                logger.info(f"Found {len(elements)} elements for selector: {selector}")

                for elem in elements[:50]:  # Limit to avoid excessive clicking
                    try:
                        if await elem.is_visible():
                            await elem.click(timeout=500, force=True)
                            await page.wait_for_timeout(200)
                    except Exception as e:
                        # Ignore click failures
                        pass

            # Wait for any animations/transitions
            await page.wait_for_timeout(2000)

            # Strategy 2: Extract all links from navigation areas
            logger.info("Extracting links from navigation...")

            nav_selectors = [
                'nav a[href]',
                'aside a[href]',
                '[class*="sidebar"] a[href]',
                '[class*="navigation"] a[href]',
                '[class*="menu"] a[href]',
                '[class*="toc"] a[href]',
                '[role="navigation"] a[href]'
            ]

            for selector in nav_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for elem in elements:
                        href = await elem.get_attribute('href')
                        if href:
                            # Make absolute URL
                            full_url = page.url if href.startswith('#') else href
                            if not href.startswith('http'):
                                from urllib.parse import urljoin
                                full_url = urljoin(start_url, href)

                            # Filter for same domain
                            parsed = urlparse(full_url)
                            if parsed.netloc == base_domain and not '#' in full_url:
                                # Remove query params
                                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                                urls.add(clean_url)
                except Exception as e:
                    logger.debug(f"Error extracting from {selector}: {e}")

            # Strategy 3: Execute JavaScript to find all links
            logger.info("Running JavaScript to extract all document links...")

            js_urls = await page.evaluate('''() => {
                const urls = new Set();
                const baseHost = window.location.hostname;

                // Get all links
                document.querySelectorAll('a[href]').forEach(link => {
                    try {
                        const url = new URL(link.href, window.location.origin);

                        // Filter for same domain, no fragments
                        if (url.hostname === baseHost && !link.href.includes('#')) {
                            // Remove query params
                            const cleanUrl = url.origin + url.pathname;
                            urls.add(cleanUrl);
                        }
                    } catch (e) {
                        // Invalid URL, skip
                    }
                });

                return Array.from(urls);
            }''')

            urls.update(js_urls)
            logger.info(f"JavaScript extraction found {len(js_urls)} URLs")

            # Strategy 4: Try scrolling to trigger lazy-loaded content
            logger.info("Scrolling to trigger lazy-loaded navigation...")

            await page.evaluate('''() => {
                window.scrollTo(0, document.body.scrollHeight);
            }''')
            await page.wait_for_timeout(1000)

            # Extract again after scrolling
            js_urls_after = await page.evaluate('''() => {
                const urls = new Set();
                const baseHost = window.location.hostname;

                document.querySelectorAll('a[href]').forEach(link => {
                    try {
                        const url = new URL(link.href, window.location.origin);
                        if (url.hostname === baseHost && !link.href.includes('#')) {
                            const cleanUrl = url.origin + url.pathname;
                            urls.add(cleanUrl);
                        }
                    } catch (e) {}
                });

                return Array.from(urls);
            }''')

            urls.update(js_urls_after)

        except Exception as e:
            logger.error(f"Error during Archbee discovery: {e}")
        finally:
            await browser.close()

    # Filter and limit
    filtered_urls = [url for url in urls if url.startswith('http')]
    filtered_urls = sorted(list(set(filtered_urls)))[:max_urls]

    logger.info(f"Discovered {len(filtered_urls)} unique URLs from Archbee site")

    return filtered_urls
