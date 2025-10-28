"""Documentation scrapers for DocRAG"""

from .base import BaseScraper
from .generic import GenericDocScraper, scrape_url

# Import Crawl4AI scraper if available
try:
    from .crawl4ai_scraper import Crawl4AIScraper, scrape_url_smart, CRAWL4AI_AVAILABLE
    __all__ = [
        "BaseScraper",
        "GenericDocScraper",
        "scrape_url",
        "Crawl4AIScraper",
        "scrape_url_smart",
        "CRAWL4AI_AVAILABLE"
    ]
except ImportError:
    CRAWL4AI_AVAILABLE = False
    __all__ = ["BaseScraper", "GenericDocScraper", "scrape_url", "CRAWL4AI_AVAILABLE"]
