#!/bin/bash

# Script to fix mypy issues in the codebase
# This script applies multiple fixes to address common mypy errors

echo "🔍 Starting mypy issue fixing process..."

# 1. Apply type annotations to variables
echo "📝 Adding basic type annotations to variables..."
python3 scripts/add_basic_type_annotations.py

# 2. Fix Path vs str compatibility issues
echo "🛠️ Fixing Path vs str compatibility issues..."
python3 scripts/fix_path_vs_str.py

# 3. Run pre-commit with the new mypy configuration
echo "🧪 Running pre-commit to check if mypy issues are resolved..."
pre-commit run mypy

echo "✅ Fix process completed!"
echo 
echo "If there are still mypy errors:"
echo "1. Add specific type annotations to your code"
echo "2. Add module-specific overrides in pyproject.toml if needed"
echo "3. Run 'mypy --show-error-codes' to identify specific issues"
echo
echo "For a long-term solution, consider gradually improving type safety in your codebase." 