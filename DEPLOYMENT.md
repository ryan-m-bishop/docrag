# DocRAG Deployment & Update Workflow

Complete guide for deploying DocRAG across multiple machines and keeping them synchronized.

## Summary of Changes

This deployment workflow was created to address:
1. **UTF-8 Syntax Error** in `docrag/server.py:134` - Fixed invalid byte sequences
2. **Easy Updates** - Created automated update scripts and workflows
3. **Multi-Machine Sync** - Documented process for keeping installations in sync

## Files Created

- `update.sh` - Automated update script with installation detection
- `Makefile` - Convenient make targets for common operations
- `INSTALL.md` - Step-by-step installation guide for new machines
- `UPDATE_GUIDE.md` - Comprehensive update workflow documentation
- `DEPLOYMENT.md` - This file - complete deployment overview

## Deployment Strategy

### Philosophy

DocRAG uses **editable installations** (`pipx install -e .`) for easy updates:
- Changes pulled via git are immediately active
- No reinstallation required for code changes
- Perfect for homelab/development environments
- Consistent behavior across all machines

### Architecture

```
Git Repository (main branch)
    ↓
    git push
    ↓
Multiple Machines
    ↓
    git pull (automatic with editable install)
    ↓
    Updated Code (no reinstall needed)
```

## Quick Reference

### On Development Machine (Making Changes)

```bash
# 1. Make changes
vim docrag/server.py

# 2. Test locally (if editable install)
docrag --help  # changes already active

# 3. Commit and push
git add .
git commit -m "Description of changes"
git push origin main
```

### On Other Machines (Receiving Updates)

```bash
# Method 1: Automated (recommended)
cd /opt/claude-ops/doc-rag
./update.sh

# Method 2: Manual (if editable install)
cd /opt/claude-ops/doc-rag
git pull

# Method 3: Using Make
cd /opt/claude-ops/doc-rag
make update
```

## Initial Deployment to New Machine

### Prerequisites
```bash
# Ensure prerequisites are met
python3 --version  # Should be 3.10+
pipx --version     # Install if needed: python3 -m pip install --user pipx
git --version
```

### Installation Steps

```bash
# 1. Clone repository
cd /opt/claude-ops
git clone <your-repo-url> doc-rag
cd doc-rag

# 2. Install with pipx (editable mode)
pipx install -e .

# 3. Initialize
docrag init

# 4. Configure Claude Code (optional)
# Edit ~/.config/claude-code/mcp_settings.json
# Add docrag server configuration

# 5. Verify
docrag --version
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Update Workflow

### Automatic Updates (Editable Install)

If installed with `pipx install -e .`:

```bash
cd /opt/claude-ops/doc-rag
git pull
# Done! Changes are already active
```

### Using Update Script (All Installations)

The `update.sh` script handles both editable and regular installs:

```bash
cd /opt/claude-ops/doc-rag
./update.sh
```

**What it does:**
1. Fetches updates from git
2. Pulls latest changes
3. Detects installation type (pipx/pip, editable/regular)
4. Reinstalls only if necessary
5. Reports status

### Verification

```bash
# Check git version
git log -1 --oneline

# Test installation
docrag --version
docrag --help

# Test functionality
docrag list
```

## Maintenance Tasks

### Update All Machines (Scripted)

Create a script to update multiple machines via SSH:

```bash
#!/bin/bash
# update-all-machines.sh

MACHINES=(
    "user@machine1.local"
    "user@machine2.local"
    "user@machine3.local"
)

for machine in "${MACHINES[@]}"; do
    echo "Updating $machine..."
    ssh "$machine" "cd /opt/claude-ops/doc-rag && git pull"
    echo "✓ $machine updated"
done
```

### Rollback to Previous Version

```bash
cd /opt/claude-ops/doc-rag

# View recent commits
git log --oneline -10

# Rollback to specific commit
git reset --hard <commit-hash>

# Or rollback one commit
git reset --hard HEAD~1

# For editable install, changes are immediate
# For regular install, reinstall:
pipx uninstall docrag && pipx install -e .
```

### Check Installation Status

```bash
# Check if installed
pipx list | grep docrag

# Check installation type
pipx environment --value PIPX_LOCAL_VENVS
ls -la ~/.local/pipx/venvs/docrag/lib/python*/site-packages/

# Check git status
cd /opt/claude-ops/doc-rag
git status
git log -1
```

## Troubleshooting

### Changes Not Reflecting After Update

**Cause**: Non-editable install

**Solution**:
```bash
pipx uninstall docrag
pipx install -e /opt/claude-ops/doc-rag
```

### Git Conflicts During Pull

**Solution**:
```bash
# Stash local changes
git stash

# Pull updates
git pull

# Review and reapply changes
git stash pop
```

### Permission Errors

**Solution**:
```bash
# Fix ownership
sudo chown -R $USER:$USER /opt/claude-ops/doc-rag

# Or use user directory
mkdir -p ~/projects
git clone <url> ~/projects/doc-rag
pipx install -e ~/projects/doc-rag
```

### Import Errors After Update

**Cause**: Dependencies changed

**Solution**:
```bash
cd /opt/claude-ops/doc-rag
pipx uninstall docrag
pipx install -e .
```

## Best Practices

1. **Always Use Editable Installs** for development/homelab environments
2. **Test Updates on One Machine First** before deploying to all
3. **Keep Git Clean** - commit or stash changes before pulling
4. **Document Custom Changes** in git commits
5. **Backup Configuration** (`~/.docrag/`) before major updates
6. **Use Update Script** rather than manual commands
7. **Verify After Updates** with `docrag --version` and basic tests

## CI/CD Considerations (Future)

For production deployments, consider:

1. **GitHub Actions** for automated testing
2. **Version Tags** for stable releases
3. **Semantic Versioning** in `pyproject.toml`
4. **PyPI Publishing** for pip install from package index
5. **Docker Containers** for isolated deployments

## Monitoring

### Health Check Command

```bash
# Quick health check
docrag list && echo "✓ DocRAG is healthy"
```

### Log Locations

- **Application logs**: Printed to stdout during `docrag serve`
- **Git logs**: `git log` in repo directory
- **Installation logs**: `pipx list --verbose`

## Security Notes

1. **Repository Access**: Ensure git credentials are configured securely
2. **File Permissions**: Check `/opt/claude-ops/doc-rag` permissions
3. **Updates**: Review changes before deploying to production systems
4. **API Keys**: Use environment variables, not hardcoded values

## Support

- **Installation Issues**: See [INSTALL.md](INSTALL.md)
- **Update Issues**: See [UPDATE_GUIDE.md](UPDATE_GUIDE.md)
- **Usage Questions**: See [README.md](README.md)
- **Bug Reports**: Check git issues or create new ones

## Changelog

### 2025-01-03
- Fixed UTF-8 syntax error in `server.py:134`
- Created update.sh automated update script
- Added Makefile for common operations
- Documented deployment and update workflows
- Created INSTALL.md for new machine setup
