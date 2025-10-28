"""Test Crawl4AI integration"""

import asyncio
from pathlib import Path
import tempfile
import sys


async def test_smart_scraping():
    """Test that Crawl4AI scraping works"""

    print("Testing Crawl4AI integration...")

    try:
        from docrag.scrapers.crawl4ai_scraper import Crawl4AIScraper, CRAWL4AI_AVAILABLE

        if not CRAWL4AI_AVAILABLE:
            print("ERROR: Crawl4AI not available")
            print("Install with: pip install crawl4ai")
            return False

        print("SUCCESS: Crawl4AI is available")

        # Test with a simple page (no LLM needed)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            scraper = Crawl4AIScraper(
                output_dir=output_dir,
                max_pages=1,
                use_llm=False  # Don't require API key for test
            )

            # Test with a simple documentation page
            test_url = "https://docs.python.org/3/library/pathlib.html"

            print(f"Testing scrape of: {test_url}")
            pages = await scraper.scrape(test_url)

            if pages > 0:
                print(f"SUCCESS: Successfully scraped {pages} page(s)")

                # Check that content was saved
                files = list(output_dir.glob("*.md"))
                if files:
                    print(f"SUCCESS: Created {len(files)} markdown file(s)")

                    # Check content
                    content = files[0].read_text()
                    if len(content) > 100:
                        print(f"SUCCESS: Content length: {len(content)} characters")
                        return True
                    else:
                        print(f"ERROR: Content too short: {len(content)} characters")
                        return False
                else:
                    print("ERROR: No files created")
                    return False
            else:
                print("ERROR: No pages scraped")
                return False

    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("Install Crawl4AI with: pip install crawl4ai")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_smart_scraping())
    sys.exit(0 if result else 1)
