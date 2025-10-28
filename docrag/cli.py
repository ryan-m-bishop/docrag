import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from pathlib import Path
import sys
import asyncio
import json
import shutil

from .config import ConfigManager, CollectionMetadata
from .embeddings import EmbeddingGenerator
from .vectordb import VectorDB
from .indexer import DocumentIndexer

console = Console()


@click.group()
def main():
    """DocRAG - Documentation RAG for AI Coding Assistants"""
    pass


@main.command()
def init():
    """Initialize DocRAG configuration"""
    config_manager = ConfigManager()

    if config_manager.config_file.exists():
        console.print("WARNING: DocRAG is already initialized", style="yellow")
        return

    with console.status("[bold green]Initializing DocRAG..."):
        config_manager.init()

    console.print("SUCCESS: DocRAG initialized successfully!", style="bold green")
    console.print(f"Configuration directory: {config_manager.base_path}")


@main.command()
@click.argument('name')
@click.option('--source', '-s', type=click.Path(exists=True), help='Source directory or URL')
@click.option('--description', '-d', help='Collection description')
def add(name: str, source: str, description: str):
    """Add a new documentation collection"""
    config_manager = ConfigManager()

    if not config_manager.config_file.exists():
        console.print("ERROR: DocRAG not initialized. Run 'docrag init' first.", style="bold red")
        sys.exit(1)

    collection_dir = config_manager.collections_dir / name
    if collection_dir.exists():
        console.print(f"ERROR: Collection '{name}' already exists", style="bold red")
        sys.exit(1)

    # Create collection directory
    collection_dir.mkdir(parents=True)
    source_docs_dir = collection_dir / "source_docs"
    source_docs_dir.mkdir()

    # Save metadata
    metadata = CollectionMetadata(
        name=name,
        source_type='local',
        source_path=source,
        description=description
    )

    with open(collection_dir / "metadata.json", 'w') as f:
        f.write(metadata.model_dump_json(indent=2))

    console.print(f"SUCCESS: Created collection '{name}'", style="bold green")

    if source:
        console.print(f"\nIndexing documents from: {source}")
        _index_collection(name, Path(source), config_manager)


def _index_collection(name: str, source_dir: Path, config_manager: ConfigManager):
    """Index documents in a collection"""
    embeddings = EmbeddingGenerator()
    vectordb = VectorDB(config_manager.vectordb_dir)
    indexer = DocumentIndexer(embeddings, vectordb)

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Indexing {name}...", total=None)

        total_chunks = indexer.index_directory(name, source_dir)

        progress.update(task, completed=100)

    console.print(f"SUCCESS: Indexed {total_chunks} chunks", style="bold green")

    # Update config to activate collection
    config = config_manager.load_config()
    if name not in config.active_collections:
        config.active_collections.append(name)
        config_manager.save_config(config)


@main.command()
@click.argument('name')
def remove(name: str):
    """Remove a documentation collection"""
    config_manager = ConfigManager()
    vectordb = VectorDB(config_manager.vectordb_dir)

    if not click.confirm(f"Are you sure you want to remove collection '{name}'?"):
        return

    # Remove from vector DB
    vectordb.delete_collection(name)

    # Remove collection directory
    collection_dir = config_manager.collections_dir / name
    if collection_dir.exists():
        shutil.rmtree(collection_dir)

    # Update config
    config = config_manager.load_config()
    if name in config.active_collections:
        config.active_collections.remove(name)
        config_manager.save_config(config)

    console.print(f"SUCCESS: Removed collection '{name}'", style="bold green")


@main.command()
def list():
    """List all documentation collections"""
    config_manager = ConfigManager()
    vectordb = VectorDB(config_manager.vectordb_dir)

    collections = vectordb.list_collections()

    if not collections:
        console.print("No collections found. Use 'docrag add <name>' to create one.", style="yellow")
        return

    config = config_manager.load_config()

    table = Table(title="Documentation Collections")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    for coll in collections:
        status = "Active" if coll in config.active_collections else "Inactive"

        # Load metadata if available
        metadata_file = config_manager.collections_dir / coll / "metadata.json"
        description = ""
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
                description = meta.get('description', '')

        table.add_row(coll, status, description)

    console.print(table)


@main.command()
@click.argument('name')
@click.argument('source', type=click.Path(exists=True))
def update(name: str, source: str):
    """Update a collection with new documents"""
    config_manager = ConfigManager()

    collection_dir = config_manager.collections_dir / name
    if not collection_dir.exists():
        console.print(f"ERROR: Collection '{name}' not found", style="bold red")
        sys.exit(1)

    console.print(f"Updating collection '{name}'...")
    _index_collection(name, Path(source), config_manager)


@main.command()
def serve():
    """Start the MCP server"""
    from .server import main as server_main

    console.print("Starting DocRAG MCP Server...", style="bold green")
    console.print("Listening on stdio for Claude Code connection...\n")

    asyncio.run(server_main())


@main.command()
@click.argument('query')
@click.option('--collection', '-c', help='Specific collection to search')
@click.option('--limit', '-l', default=5, help='Number of results')
def search(query: str, collection: str, limit: int):
    """Search documentation from CLI (for testing)"""
    config_manager = ConfigManager()
    embeddings = EmbeddingGenerator()
    vectordb = VectorDB(config_manager.vectordb_dir)

    # Generate query embedding
    console.print(f"[cyan]Searching for:[/cyan] {query}\n")
    query_embedding = embeddings.embed_single(query)

    # Determine collections to search
    if collection:
        collections = [collection]
    else:
        config = config_manager.load_config()
        collections = config.active_collections or vectordb.list_collections()

    # Search
    all_results = []
    for coll in collections:
        results = vectordb.search(coll, query_embedding.tolist(), limit)
        for result in results:
            result['collection'] = coll
        all_results.extend(results)

    # Display results
    if not all_results:
        console.print(f"No results found for: {query}", style="yellow")
        return

    console.print(f"[bold]Search Results:[/bold]\n")

    for i, result in enumerate(all_results[:limit], 1):
        console.print(f"[bold cyan]Result {i}[/bold cyan]")
        console.print(f"Collection: {result.get('collection')}")
        console.print(f"Source: {result.get('metadata', {}).get('source', 'unknown')}")
        console.print(f"Score: {1 - result.get('_distance', 0):.3f}")
        console.print(f"\n{result.get('text', '')}\n")
        console.print("-" * 80)


@main.command()
@click.argument('url')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output directory')
@click.option('--playwright', is_flag=True, help='Use Playwright for dynamic content (basic scraper)')
@click.option('--smart', '--use-crawl4ai', 'use_smart', is_flag=True, help='Use AI-powered Crawl4AI scraper (recommended)')
@click.option('--no-llm', is_flag=True, help='Disable LLM extraction in Crawl4AI (faster, less accurate)')
@click.option('--llm-provider', default='openai/gpt-4o-mini', help='LLM provider for smart scraping')
@click.option('--max-pages', default=1000, type=int, help='Maximum pages to scrape')
def scrape(url: str, output: str, playwright: bool, use_smart: bool, no_llm: bool, llm_provider: str, max_pages: int):
    """Scrape documentation from a website

    \b
    Basic scraping (fast, generic):
      docrag scrape https://docs.example.com --output ./docs

    \b
    Smart scraping with AI (recommended, better quality):
      docrag scrape https://docs.example.com --output ./docs --smart

    \b
    Smart scraping without LLM (faster, still better than basic):
      docrag scrape https://docs.example.com --output ./docs --smart --no-llm

    \b
    For JavaScript-heavy sites (basic scraper):
      docrag scrape https://example.com --output ./docs --playwright
    """
    from rich.panel import Panel
    from docrag.scrapers import CRAWL4AI_AVAILABLE

    output_dir = Path(output)

    # Check if smart scraping is requested but not available
    if use_smart and not CRAWL4AI_AVAILABLE:
        console.print()
        console.print(Panel.fit(
            "[yellow]WARNING: Crawl4AI not installed[/]\n\n"
            "Smart scraping requires Crawl4AI. Install it with:\n"
            "  [cyan]pip install crawl4ai[/]\n\n"
            "Or install DocRAG with smart scraping support:\n"
            "  [cyan]pipx inject docrag crawl4ai[/]\n\n"
            "Falling back to basic scraper...",
            title="Smart Scraping Unavailable",
            border_style="yellow"
        ))
        use_smart = False

    console.print()
    console.print(f"[bold]Scraping documentation[/]")
    console.print(f"  URL: [cyan]{url}[/]")
    console.print(f"  Output: [cyan]{output_dir}[/]")

    if use_smart:
        console.print(f"  Method: [green]Smart (Crawl4AI)[/]")
        if not no_llm:
            console.print(f"  LLM: [yellow]{llm_provider}[/]")
            console.print(f"  [dim]Set OPENAI_API_KEY environment variable[/]")
        else:
            console.print(f"  LLM: [dim]Disabled[/]")
    else:
        console.print(f"  Method: [yellow]{'Playwright' if playwright else 'Basic'}[/]")

    console.print(f"  Max pages: [yellow]{max_pages}[/]")
    console.print()

    async def run():
        if use_smart:
            from docrag.scrapers.crawl4ai_scraper import scrape_url_smart
            return await scrape_url_smart(
                url,
                output_dir,
                max_pages=max_pages,
                use_llm=not no_llm,
                llm_provider=llm_provider
            )
        else:
            from docrag.scrapers.generic import scrape_url
            return await scrape_url(url, output_dir, playwright, max_pages)

    try:
        with console.status("[cyan]Scraping in progress..."):
            pages = asyncio.run(run())

        console.print(f"\nSUCCESS: Scraped [green]{pages}[/] pages to [cyan]{output_dir}[/]", style="bold")
        console.print(f"\nNext step: [yellow]docrag add <name> --source {output_dir}[/]")

    except Exception as e:
        console.print(f"\nERROR: {e}", style="bold red")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        sys.exit(1)


if __name__ == '__main__':
    main()
