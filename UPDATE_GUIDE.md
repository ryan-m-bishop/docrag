# DocRAG Update Guide

Quick reference for keeping DocRAG up to date across multiple machines.

## Quick Update Commands

### For Machines with Editable Install

If you installed with `pipx install -e .`, updates are automatic:

```bash
cd /opt/claude-ops/doc-rag
git pull
```

That's it! No reinstall needed.

### For All Installations (Universal)

Use the update script:

```bash
cd /opt/claude-ops/doc-rag
./update.sh
```

Or use Make:

```bash
cd /opt/claude-ops/doc-rag
make update
```

## Initial Setup on New Machines

### 1. Clone the Repository

```bash
cd /opt/claude-ops
git clone <your-repo-url> doc-rag
cd doc-rag
```

### 2. Install with pipx (Recommended)

```bash
# Install in editable mode for automatic updates
pipx install -e .

# Verify
docrag --version
```

### 3. Configure for Claude Code

Add to `~/.config/claude-code/mcp_settings.json`:

```json
{
  "mcpServers": {
    "docrag": {
      "command": "docrag",
      "args": ["serve"]
    }
  }
}
```

## Update Workflow

### For the Developer (Pushing Changes)

```bash
# 1. Make your changes
vim docrag/server.py

# 2. Test locally
docrag --help

# 3. Commit and push
git add .
git commit -m "Fix UTF-8 encoding issue in server.py"
git push origin main
```

### For Other Machines (Pulling Changes)

```bash
# Method 1: Use the update script
cd /opt/claude-ops/doc-rag
./update.sh

# Method 2: Manual (if editable install)
cd /opt/claude-ops/doc-rag
git pull

# Method 3: Use Make
cd /opt/claude-ops/doc-rag
make update
```

## Understanding Installation Types

### Editable Install (`-e` flag)
- Changes to source code take effect immediately
- No reinstall needed after git pull
- Recommended for development and frequently updated systems
- Install command: `pipx install -e .`

### Regular Install (no `-e` flag)
- Creates a snapshot of the code
- Requires reinstall after updates
- Better for production/stable systems
- Install command: `pipx install .`

## Troubleshooting

### "Command not found: docrag"

Check installation:
```bash
pipx list | grep docrag
# or
pip show docrag
```

Reinstall if needed:
```bash
pipx install -e /opt/claude-ops/doc-rag
```

### "Changes not reflected after git pull"

You might have a regular install (not editable). Reinstall:
```bash
pipx reinstall docrag --force
```

Or switch to editable:
```bash
pipx uninstall docrag
pipx install -e /opt/claude-ops/doc-rag
```

### "Git conflicts during pull"

```bash
# Stash your changes
git stash

# Pull updates
git pull

# Reapply your changes
git stash pop
```

## Best Practices

1. **Always use editable installs** during active development
2. **Run the update script** rather than manual commands
3. **Test after updates** with `docrag --help` or `docrag list`
4. **Keep a backup** of your `~/.docrag/` data directory
5. **Document custom changes** if you modify the code

## Version Information

Check current version and git commit:
```bash
# Python package version
docrag --version

# Git commit
cd /opt/claude-ops/doc-rag
git log -1 --oneline

# Git branch
git branch --show-current
```
