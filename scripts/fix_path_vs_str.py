#!/usr/bin/env python3
"""
Script to fix Path vs str compatibility issues in the codebase.
This adds appropriate str() conversions to Path objects where needed.
"""

import os
import re
import sys
from pathlib import Path

# Patterns to match Path vs str issues
PATH_RETURN_PATTERN = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s*->\s*str\s*:')
PATH_ARG_PATTERN = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*')

def process_file(file_path):
    """Process a file to fix Path vs str issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Keep track of whether we modified the file
        modified = False
        
        # Track functions that return Path but are annotated to return str
        return_matches = list(PATH_RETURN_PATTERN.finditer(content))
        for match in return_matches:
            func_name = match.group(1)
            
            # Find the return statements in this function
            func_start = match.end()
            # Look for "return Path" pattern and replace with "return str(Path)"
            return_pattern = re.compile(r'(\s+return\s+)(Path\([^)]+\))')
            content_after = content[func_start:]
            
            # Find the next function definition or end of file
            next_func_match = re.search(r'\ndef\s+', content_after)
            if next_func_match:
                func_end = func_start + next_func_match.start()
                func_content = content[func_start:func_end]
            else:
                func_content = content_after
            
            modified_func_content = re.sub(
                return_pattern, 
                r'\1str(\2)', 
                func_content
            )
            
            if modified_func_content != func_content:
                content = content[:func_start] + modified_func_content + content[func_start + len(func_content):]
                modified = True
        
        # Fix function calls with Path arguments to str parameters
        # This is more complex and may require manual fixing
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Path vs str issues in {file_path}")
            return True
        return False
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory, extensions=('.py',), skip_dirs=('venv', '.venv', '.git')):
    """Process all Python files in a directory recursively."""
    modified_count = 0
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip specified directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                file_count += 1
                if process_file(file_path):
                    modified_count += 1
    
    print(f"Processed {file_count} files, modified {modified_count} files")
    return modified_count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.abspath('.')
    
    print(f"Fixing Path vs str issues in Python files in {directory}")
    process_directory(directory) 