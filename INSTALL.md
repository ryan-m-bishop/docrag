# DocRAG Installation Guide

Quick installation guide for setting up DocRAG on new machines.

## One-Line Install (Recommended)

For machines with the repo already at `/opt/claude-ops/doc-rag`:

```bash
cd /opt/claude-ops/doc-rag && pipx install -e . && echo "✓ DocRAG installed!"
```

## Step-by-Step Installation

### 1. Prerequisites

```bash
# Install pipx (if not already installed)
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Restart shell or run:
source ~/.bashrc  # or ~/.zshrc
```

### 2. Clone Repository (if needed)

```bash
cd /opt/claude-ops
git clone <your-repo-url> doc-rag
cd doc-rag
```

### 3. Install DocRAG

```bash
# Install in editable mode (recommended - updates automatically with git pull)
pipx install -e .

# Verify installation
docrag --version
docrag --help
```

### 4. Initialize DocRAG

```bash
# Create configuration directory and initialize
docrag init
```

### 5. Optional: Install Playwright for Web Scraping

```bash
# Install playwright in the pipx environment
pipx runpip docrag install playwright

# Install browser
pipx run --spec docrag playwright install chromium
```

## Configure Claude Code Integration

### 1. Edit MCP Settings

```bash
# Open or create the MCP settings file
nano ~/.config/claude-code/mcp_settings.json
```

### 2. Add DocRAG Server

Add this configuration:

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

Or if you need the full path:

```json
{
  "mcpServers": {
    "docrag": {
      "command": "/home/YOUR_USERNAME/.local/bin/docrag",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

### 3. Restart Claude Code

Close and reopen Claude Code to load the MCP server.

## Verify Installation

```bash
# Check DocRAG is installed
docrag --version

# Check configuration
docrag list

# Test search (if you have collections)
docrag search "test query"
```

## Quick Start

```bash
# Add a documentation collection
docrag add myproject --source ~/docs/myproject --description "My project documentation"

# List collections
docrag list

# Search
docrag search "how to configure"

# Start MCP server (usually done automatically by Claude Code)
docrag serve
```

## Installation Locations

- **Binary**: `~/.local/bin/docrag` (added to PATH by pipx)
- **Configuration**: `~/.docrag/`
- **Source code**: `/opt/claude-ops/doc-rag/` (if editable install)

## Uninstall

```bash
# Remove DocRAG
pipx uninstall docrag

# Optionally remove configuration
rm -rf ~/.docrag
```

## Troubleshooting

### "Command not found: docrag"

Ensure pipx bin directory is in your PATH:
```bash
pipx ensurepath
source ~/.bashrc  # or ~/.zshrc
```

### Permission Issues

If you can't write to `/opt/claude-ops/`, install from a user directory:
```bash
mkdir -p ~/projects
cd ~/projects
git clone <your-repo-url> doc-rag
cd doc-rag
pipx install -e .
```

### Import Errors

Reinstall with force:
```bash
pipx uninstall docrag
pipx install -e /opt/claude-ops/doc-rag
```

## Next Steps

- See [README.md](README.md) for usage documentation
- See [UPDATE_GUIDE.md](UPDATE_GUIDE.md) for updating instructions
- Run `docrag --help` for all available commands
