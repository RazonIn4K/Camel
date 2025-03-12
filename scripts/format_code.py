#!/usr/bin/env python3
"""Script to automatically format and clean Python code in the project."""
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(command: List[str]) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def check_dependencies() -> Dict[str, bool]:
    """Verify that required tools are installed."""
    required_tools = [
        "black",
        "autoflake",
        "isort",
        "mypy",
        "docformatter",
    ]
    results = {}
    for tool in required_tools:
        success, _ = run_command([tool, "--version"])
        results[tool] = success
        if not success:
            print(f"Error: {tool} is not installed. Please install it with:")
            print(f"pip install {tool}")
    return results


def format_project() -> bool:
    """Format all Python files in the project."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.glob("**/*.py"))
    python_files_str = [str(f) for f in python_files]

    # Run black formatter
    print("Running black formatter...")
    success, output = run_command(
        ["black", "--line-length", "88", "--target-version", "py37", *python_files_str]
    )
    if not success:
        print("Error running black:", output)
        return False
    print("Black formatting complete.")

    # Run isort
    print("\nRunning isort to sort imports...")
    success, output = run_command(
        [
            "isort",
            "--profile",
            "black",
            "--multi-line",
            "3",
            "--line-length",
            "88",
            *python_files_str,
        ]
    )
    if not success:
        print("Error running isort:", output)
        return False
    print("Import sorting complete.")

    # Run autoflake
    print("\nRunning autoflake to remove unused imports...")
    success, output = run_command(
        [
            "autoflake",
            "--in-place",
            "--recursive",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--expand-star-imports",
            *python_files_str,
        ]
    )
    if not success:
        print("Error running autoflake:", output)
        return False
    print("Autoflake cleanup complete.")

    # Run docformatter
    print("\nRunning docformatter...")
    success, output = run_command(
        [
            "docformatter",
            "--in-place",
            "--recursive",
            "--wrap-summaries",
            "88",
            "--wrap-descriptions",
            "88",
            *python_files_str,
        ]
    )
    if not success:
        print("Error running docformatter:", output)
        return False
    print("Docstring formatting complete.")

    # Run mypy for type checking
    print("\nRunning mypy type checker...")
    success, output = run_command(
        ["mypy", "--ignore-missing-imports", "--strict", *python_files_str]
    )
    if not success:
        print("Type checking issues found:", output)
        # Don't return False here as type issues might need manual fixing
    print("Type checking complete.")

    return True


def setup_pre_commit() -> bool:
    """Set up pre-commit hook configuration."""
    try:
        # First check if we're in a git repository
        success, output = run_command(['git', 'rev-parse', '--git-dir'])
        if not success:
            print("Error: Not in a git repository. Please run 'git init' first.")
            return False

        pre_commit_config = """
repos:
-   repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
    -   id: black
        args: [--line-length=88, --target-version=py37]
-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: [--profile=black, --line-length=88]
-   repo: https://github.com/PyCQA/autoflake
    rev: v2.0.0
    hooks:
    -   id: autoflake
        args: [--remove-all-unused-imports, --remove-unused-variables]
-   repo: https://github.com/mypy/mypy
    rev: v1.0.0
    hooks:
    -   id: mypy
        args: [--ignore-missing-imports, --strict]
"""
        with open('.pre-commit-config.yaml', 'w') as f:
            f.write(pre_commit_config)
        
        print("Installing pre-commit hooks...")
        success, output = run_command(['pre-commit', 'install'])
        if not success:
            print(f"Error installing pre-commit hook: {output}")
            return False
        
        print("Pre-commit hooks installed successfully")
        return True
    except Exception as e:
        print(f"Error setting up pre-commit: {str(e)}")
        return False


def main() -> int:
    """Main entry point."""
    print("Checking dependencies...")
    dep_results = check_dependencies()
    if not all(dep_results.values()):
        print("\nPlease install missing dependencies and try again.")
        return 1

    print("\nStarting code formatting...")
    if not format_project():
        return 1

    print("\nSetting up pre-commit hooks...")
    if not setup_pre_commit():
        print("Warning: Pre-commit setup failed, but formatting completed")

    print("\nCode formatting completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
