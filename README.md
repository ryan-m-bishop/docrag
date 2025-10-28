# DocRAG - AI Documentation RAG System

A lightweight, installable Python package that provides RAG (Retrieval Augmented Generation) access to technical documentation through an MCP (Model Context Protocol) server. This enables Claude Code to search and retrieve relevant documentation on-demand.

## Features

- 🚀 Single pip-installable package with CLI and MCP server
- 📚 Project-based documentation collections (BrightSign, Venafi, Qumu, web frameworks)
- 🔍 Local vector database with efficient embedding using LanceDB
- 📥 Easy documentation ingestion from local files or scraped sources
- 🤖 Designed for use with Claude Code via MCP

## Installation

### Prerequisites

- Python 3.10+
- pipx (recommended) or pip

### Recommended: Install globally with pipx

```bash
# Install globally with pipx (keeps dependencies isolated)
pipx install -e /opt/claude-ops/doc-rag

# Verify installation
docrag --help

# Optional: Install Playwright browsers (for scraping)
pipx runpip docrag install playwright
pipx run --spec docrag playwright install chromium
```

### Alternative: Install from source (development)

```bash
# Clone or navigate to the project directory
cd /opt/claude-ops/doc-rag

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install Playwright browsers (for scraping)
playwright install chromium
```

## Quick Start

### 1. Initialize DocRAG

```bash
docrag init
```

This creates the configuration directory at `~/.docrag/` with the following structure:

```
~/.docrag/
├── config.json           # Global configuration
├── collections/          # Documentation collections
└── vectordb/            # LanceDB storage
```

### 2. Add a Documentation Collection

```bash
# Add documentation from a local directory
docrag add brightsign --source /path/to/brightsign/docs --description "BrightSign player documentation"

# Or add without source initially
docrag add venafi --description "Venafi TPP API documentation"
```

### 3. List Collections

```bash
docrag list
```

### 4. Search Documentation (CLI Testing)

```bash
# Search across all active collections
docrag search "how to initialize the player"

# Search a specific collection
docrag search "authentication methods" --collection venafi --limit 10
```

### 5. Start the MCP Server

```bash
docrag serve
```

The server will listen on stdio for connections from Claude Code.

## CLI Commands

### `docrag init`
Initialize DocRAG configuration directory.

### `docrag add <name>`
Add a new documentation collection.

Options:
- `-s, --source PATH` - Source directory containing documentation
- `-d, --description TEXT` - Description of the collection

Example:
```bash
docrag add qumu --source ~/docs/qumu --description "Qumu video platform docs"
```

### `docrag list`
List all documentation collections with their status.

### `docrag update <name> <source>`
Update an existing collection with new documents.

Example:
```bash
docrag update brightsign ~/docs/brightsign/updated
```

### `docrag remove <name>`
Remove a documentation collection (with confirmation).

### `docrag search <query>`
Search documentation from the CLI for testing.

Options:
- `-c, --collection TEXT` - Specific collection to search
- `-l, --limit INTEGER` - Number of results (default: 5)

Example:
```bash
docrag search "websocket connection" --collection brightsign
```

### `docrag serve`
Start the MCP server for Claude Code integration.

### `docrag scrape <url>`
Scrape documentation from websites.

Options:
- `-o, --output PATH` - Output directory (required)
- `--smart, --use-crawl4ai` - Use AI-powered Crawl4AI scraper (recommended)
- `--no-llm` - Disable LLM extraction (faster, still better than basic)
- `--llm-provider TEXT` - LLM provider (default: openai/gpt-4o-mini)
- `--playwright` - Use Playwright for dynamic content (basic scraper)
- `--max-pages INTEGER` - Maximum pages to scrape (default: 1000)

Examples:
```bash
# Basic scraping
docrag scrape https://docs.example.com --output ./docs

# Smart scraping with AI (recommended)
docrag scrape https://docs.example.com --output ./docs --smart

# Smart scraping without LLM (faster, no API key needed)
docrag scrape https://docs.example.com --output ./docs --smart --no-llm

# Limit pages
docrag scrape https://docs.example.com --output ./docs --max-pages 100
```

**Smart Scraping Features:**
- ✨ AI-powered content extraction
- 🎯 Automatically removes navigation and boilerplate
- 📊 Better handling of complex layouts
- 🧠 Semantic understanding of documentation structure
- ⚡ Faster and more accurate than basic scraping

**To enable smart scraping:**
```bash
# Install Crawl4AI
pipx inject docrag crawl4ai

# Optional: Set OpenAI API key for LLM-powered extraction
export OPENAI_API_KEY='your-key-here'
```

## Using with Claude Code

### 1. Configure Claude Code MCP Settings

Add DocRAG to your Claude Code MCP configuration (`~/.config/claude-code/mcp_settings.json` or similar):

```json
{
  "mcpServers": {
    "docrag": {
      "command": "docrag",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

If using the full path:
```json
{
  "mcpServers": {
    "docrag": {
      "command": "/home/claude-admin/.local/bin/docrag",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

### 2. Restart Claude Code

After adding the configuration, restart Claude Code to load the MCP server.

### 3. Use in Claude Code

Once connected, Claude Code can use two tools:

**search_docs**: Search through indexed documentation collections
```
Query: "how to handle authentication in BrightSign"
Collection: (optional) "brightsign"
Limit: (optional) 5
```

**list_collections**: List all available documentation collections

Claude will automatically use these tools when working on projects that need documentation access.

## Architecture

### Core Components

1. **ConfigManager** (`config.py`) - Manages configuration and collection metadata
2. **EmbeddingGenerator** (`embeddings.py`) - Generates embeddings using sentence-transformers
3. **VectorDB** (`vectordb.py`) - LanceDB wrapper for vector storage and search
4. **DocumentIndexer** (`indexer.py`) - Intelligent document chunking and indexing
5. **DocRAGServer** (`server.py`) - MCP server implementation
6. **CLI** (`cli.py`) - Command-line interface

### Technical Stack

- **MCP Framework**: Official Anthropic MCP package
- **Vector Database**: LanceDB (lightweight, file-based, performant)
- **Embeddings**: sentence-transformers with all-MiniLM-L6-v2 model (384 dims, fast, local)
- **Text Processing**: langchain-text-splitters for intelligent chunking
- **CLI**: Click for user-friendly commands
- **Web Scraping**: Playwright + BeautifulSoup4 for scraping

## Data Structure

```
~/.docrag/
├── config.json                 # Global configuration
│   └── {
│         "active_collections": ["brightsign", "venafi"],
│         "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
│         "chunk_size": 512,
│         "chunk_overlap": 50
│       }
├── collections/
│   ├── brightsign/
│   │   ├── metadata.json       # Collection metadata
│   │   └── source_docs/        # Original documents
│   ├── venafi/
│   └── qumu/
└── vectordb/
    └── lancedb/                # Vector storage (one table per collection)
```

## Configuration

Global configuration is stored in `~/.docrag/config.json`:

```json
{
  "active_collections": ["brightsign", "venafi"],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

Collection metadata is stored in `~/.docrag/collections/<name>/metadata.json`:

```json
{
  "name": "brightsign",
  "source_type": "local",
  "source_path": "/path/to/docs",
  "created_at": "2025-10-28T10:00:00",
  "updated_at": "2025-10-28T10:00:00",
  "doc_count": 150,
  "description": "BrightSign player documentation"
}
```

## Development

### Project Structure

```
docrag/
├── docrag/
│   ├── __init__.py
│   ├── cli.py              # CLI commands
│   ├── server.py           # MCP server
│   ├── indexer.py          # Document indexing
│   ├── vectordb.py         # Vector database
│   ├── embeddings.py       # Embeddings
│   ├── config.py           # Configuration
│   └── scrapers/           # Web scrapers
│       ├── __init__.py
│       ├── base.py
│       └── generic.py
├── tests/
├── pyproject.toml
├── README.md
└── DOCRAG_MVP_BUILD_GUIDE.md
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Formatting

```bash
# Format with black
black docrag/

# Lint with ruff
ruff check docrag/
```

## Troubleshooting

### "DocRAG not initialized"
Run `docrag init` first to create the configuration directory.

### "No collections found"
Add a collection with `docrag add <name> --source <path>`.

### "Model download fails"
The first time you run DocRAG, it will download the sentence-transformers model (~100MB). Ensure you have internet connectivity.

### "Playwright not installed"
If using scrapers, run `playwright install chromium`.

## Future Enhancements

- [ ] Web scraper CLI commands
- [ ] Support for more file types (PDF, HTML, RST)
- [ ] Incremental indexing (only index changed files)
- [ ] Collection activation/deactivation
- [ ] Collection statistics and health checks
- [ ] Export/import collections
- [ ] Cloud sync for collections
- [ ] Advanced search filters

## License

MIT

## Author

Ryan - Built for homelab and Claude Code integration
