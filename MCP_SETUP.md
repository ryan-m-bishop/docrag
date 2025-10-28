# DocRAG MCP Server Setup for Claude Code

## Quick Setup

DocRAG is now **installed globally** via pipx and ready to use with Claude Code.

## MCP Configuration

Add this to your Claude Code MCP settings file:

### Option 1: Simple (Recommended)
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

### Option 2: Full Path
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

## Configuration File Location

The MCP settings file is typically located at:
- `~/.config/claude-code/mcp_settings.json`

Or check the Claude Code documentation for your specific setup.

## Verify Installation

Before configuring Claude Code, verify docrag is working:

```bash
# Check command is available
which docrag
# Output: /home/claude-admin/.local/bin/docrag

# Test the CLI
docrag --help

# Check existing collections
docrag list
```

## Start Using

1. **Add the MCP configuration** (shown above)
2. **Restart Claude Code**
3. **Test the connection** by asking Claude to search documentation

## Available Tools

Once connected, Claude Code can use:

### 1. search_docs
Search through indexed documentation collections.

**Parameters:**
- `query` (required): What you're looking for
- `collection` (optional): Specific collection to search
- `limit` (optional): Number of results (default: 5)

**Example:**
```
Claude will automatically use this when you ask questions like:
"Search the brightsign documentation for authentication methods"
```

### 2. list_collections
List all available documentation collections.

**Example:**
```
"Show me what documentation collections are available"
```

## Adding Documentation

Before using with Claude Code, add your documentation:

```bash
# Add BrightSign documentation
docrag add brightsign --source ~/docs/brightsign --description "BrightSign player API docs"

# Add Venafi documentation
docrag add venafi --source ~/docs/venafi --description "Venafi TPP API docs"

# Add Qumu documentation
docrag add qumu --source ~/docs/qumu --description "Qumu video platform docs"
```

## Troubleshooting

### MCP server won't start
1. Check docrag is installed: `which docrag`
2. Test manually: `docrag serve` (press Ctrl+C to stop)
3. Check Claude Code logs for errors

### No collections found
1. Initialize docrag: `docrag init`
2. Add collections: `docrag add <name> --source <path>`
3. Verify: `docrag list`

### Search returns no results
1. Verify collection has documents: `docrag list`
2. Test search manually: `docrag search "your query"`
3. Check collection is active in `~/.docrag/config.json`

## Current Installation Status

✅ **Installed**: Global installation via pipx
✅ **Location**: `/home/claude-admin/.local/bin/docrag`
✅ **Version**: 0.1.0
✅ **Data Directory**: `~/.docrag/`
✅ **Test Collection**: Available (test_collection)

## Next Steps

1. Add your real documentation collections
2. Configure Claude Code with the MCP settings above
3. Restart Claude Code
4. Start using documentation search in your coding sessions!

---

**Last Updated**: October 28, 2025
