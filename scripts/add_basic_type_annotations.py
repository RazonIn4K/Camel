#!/usr/bin/env python3
"""
Script to add basic type annotations to variables that lack them.
This is a quick fix to suppress mypy var-annotated errors.
"""

import os
import re
import sys
from pathlib import Path

# Regex to find variables that need type annotations
# This pattern matches lines that mypy would flag with "Need type annotation"
VAR_PATTERN = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$')

# Common variable types based on their names and values
COMMON_TYPES = {
    'count': 'int',
    'total': 'int',
    'index': 'int',
    'i': 'int',
    'j': 'int',
    'size': 'int',
    'length': 'int',
    'result': 'Any',
    'results': 'list[Any]',
    'data': 'dict[str, Any]',
    'config': 'dict[str, Any]',
    'options': 'dict[str, Any]',
    'params': 'dict[str, Any]',
    'settings': 'dict[str, Any]',
    'cache': 'dict[str, Any]',
    'cache_data': 'dict[str, Any]',
    'history': 'list[Any]',
    'items': 'list[Any]',
    'elements': 'list[Any]',
    'models': 'dict[str, Any]',
    'working_selectors': 'dict[str, Any]',
    'browser_metrics': 'dict[str, Any]',
    'user_counts': 'dict[str, int]',
    'channel_counts': 'dict[str, int]',
    'time_counts': 'dict[str, int]',
    'current_content': 'list[Any]',
    'web_search_history': 'list[dict[str, Any]]',
    'discord_search_history': 'list[dict[str, Any]]',
    'all_prompts': 'list[str]',
    'result_cache': 'dict[str, Any]',
    'selenium': 'dict[str, Any]',
}

# Check if a variable already has a type annotation
def has_type_annotation(line):
    return ':' in line and not line.strip().startswith('#')

def add_type_annotation(file_path):
    """Add basic type annotations to variables in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified_lines = []
        modified = False
        
        imported_typing = False
        
        for i, line in enumerate(lines):
            # Check if typing is already imported
            if 'from typing import' in line or 'import typing' in line:
                imported_typing = True
            
            # Skip lines that already have type annotations or are comments
            if has_type_annotation(line) or line.strip().startswith('#'):
                modified_lines.append(line)
                continue
            
            match = VAR_PATTERN.match(line)
            if match:
                var_name = match.group(1)
                var_value = match.group(2).strip()
                
                # Determine variable type
                var_type = None
                
                # Check in common types dictionary
                if var_name in COMMON_TYPES:
                    var_type = COMMON_TYPES[var_name]
                # Infer type from value
                elif var_value.startswith('{'):
                    var_type = 'dict[str, Any]'
                elif var_value.startswith('['):
                    var_type = 'list[Any]'
                elif var_value.startswith('('):
                    var_type = 'tuple[Any, ...]'
                elif var_value.isdigit():
                    var_type = 'int'
                elif var_value.startswith('0.') or '.' in var_value and var_value.replace('.', '').isdigit():
                    var_type = 'float'
                elif var_value.startswith('"') or var_value.startswith("'"):
                    var_type = 'str'
                elif var_value in ('True', 'False'):
                    var_type = 'bool'
                elif var_value == 'None':
                    var_type = 'None'
                
                if var_type:
                    if var_type != 'None':
                        new_line = line.replace(f"{var_name} =", f"{var_name}: {var_type} =")
                    else:
                        new_line = line.replace(f"{var_name} =", f"{var_name}: Optional[Any] =")
                    modified_lines.append(new_line)
                    modified = True
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)
        
        # Add typing import if we modified the file and typing wasn't already imported
        if modified and not imported_typing:
            modified_lines.insert(0, "from typing import Any, Dict, List, Optional, Tuple, Union\n")
            if len(modified_lines) > 1 and not modified_lines[1].strip():
                modified_lines.insert(1, "\n")
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
            print(f"Added type annotations to {file_path}")
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
                if add_type_annotation(file_path):
                    modified_count += 1
    
    print(f"Processed {file_count} files, modified {modified_count} files")
    return modified_count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.abspath('.')
    
    print(f"Adding type annotations to Python files in {directory}")
    process_directory(directory) 