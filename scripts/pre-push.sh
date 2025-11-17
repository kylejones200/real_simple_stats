#!/bin/bash
# Pre-push hook to run CI checks locally
# This matches what GitHub Actions CI runs on push/PR

set -e

echo "🔍 Running pre-push CI checks..."
echo ""

# Check if .python-version exists
if [ ! -f .python-version ]; then
    echo "❌ Error: .python-version file not found!"
    exit 1
fi

# Run checks that match CI exactly
echo "1/3 Running ruff check..."
ruff check . || { echo "❌ Ruff check failed!"; exit 1; }

echo "2/3 Running mypy type check..."
mypy . || { echo "❌ Mypy check failed!"; exit 1; }

echo "3/3 Running pytest..."
pytest -q --maxfail=1 || { echo "❌ Tests failed!"; exit 1; }

echo ""
echo "✅ All CI checks passed! Safe to push."

