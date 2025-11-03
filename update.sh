#!/bin/bash
# DocRAG Update Script
# Updates DocRAG installation from git and reinstalls if necessary

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== DocRAG Update Script ==="
echo

# Check if git repo
if [ ! -d ".git" ]; then
    echo "Error: Not a git repository. Please clone from git first."
    exit 1
fi

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Fetch updates
echo "Fetching updates from git..."
git fetch origin

# Check if there are updates
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo $LOCAL)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "Already up to date!"
else
    echo "Updates available. Pulling changes..."
    git pull origin "$CURRENT_BRANCH"
    echo "✓ Git pull completed"
fi

echo

# Check installation method
if command -v pipx &> /dev/null && pipx list | grep -q "docrag"; then
    echo "Detected: pipx installation"

    # Check if it's an editable install
    PIPX_VENV=$(pipx environment --value PIPX_LOCAL_VENVS)/docrag
    if [ -f "$PIPX_VENV/pyvenv.cfg" ]; then
        if grep -q "doc-rag" "$PIPX_VENV/lib/python"*/site-packages/docrag.egg-link 2>/dev/null; then
            echo "Editable install detected - no reinstall needed!"
            echo "Changes are already active."
        else
            echo "Regular install detected - reinstalling..."
            pipx uninstall docrag
            pipx install -e "$SCRIPT_DIR"
            echo "✓ Reinstallation completed"
        fi
    else
        echo "Reinstalling to be safe..."
        pipx uninstall docrag
        pipx install -e "$SCRIPT_DIR"
        echo "✓ Reinstallation completed"
    fi
elif command -v pip &> /dev/null && pip show docrag &> /dev/null; then
    echo "Detected: pip installation"
    echo "Reinstalling..."
    pip install -e "$SCRIPT_DIR" --force-reinstall --no-deps
    echo "✓ Reinstallation completed"
else
    echo "Warning: DocRAG is not installed"
    echo "Would you like to install it now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if command -v pipx &> /dev/null; then
            echo "Installing with pipx..."
            pipx install -e "$SCRIPT_DIR"
        else
            echo "Installing with pip..."
            pip install -e "$SCRIPT_DIR"
        fi
        echo "✓ Installation completed"
    fi
fi

echo
echo "=== Update Complete ==="
echo "Run 'docrag --version' to verify"
