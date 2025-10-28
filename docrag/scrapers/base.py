from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import asyncio
from playwright.async_api import async_playwright, Page, Browser
import httpx
from bs4 import BeautifulSoup


class BaseScraper(ABC):
    """Base class for documentation scrapers"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def scrape(self, start_url: str) -> int:
        """Scrape documentation and return count of pages scraped"""
        pass

    async def fetch_with_playwright(self, url: str) -> str:
        """Fetch dynamic content with Playwright"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle')
            content = await page.content()
            await browser.close()
            return content

    async def fetch_static(self, url: str) -> str:
        """Fetch static content with httpx"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text

    def html_to_markdown(self, html: str) -> str:
        """Convert HTML to clean markdown"""
        from markdownify import markdownify as md
        return md(html, heading_style="ATX")

    def save_page(self, filename: str, content: str):
        """Save scraped content to file"""
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
