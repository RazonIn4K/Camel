#!/usr/bin/env python3
"""Script to automatically format and clean Python code in the project."""
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BATCH_SIZE = 50  # Process 50 files at a time


def run_command(command: List[str]) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result: Any = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def process_in_batches(files: List[str], command_prefix: List[str]) -> bool:
    """Process files in batches to avoid command line length limits."""
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i : i + BATCH_SIZE]
        command = command_prefix + batch
        success, output = run_command(command)
        if not success:
            print(f"Error processing batch: {output}")
            return False
    return True


def check_dependencies() -> Dict[str, bool]:
    """Verify that required tools are installed."""
    required_tools: list[Any] = [
        "black",
        "autoflake",
        "isort",
        "mypy",
    ]
    optional_tools: list[Any] = [
        "docformatter",
    ]
    results: list[Any] = {}

    print("Checking required tools...")
    for tool in required_tools:
        success, _ = run_command([tool, "--version"])
        results[tool] = success
        if not success:
            print(f"Error: {tool} is not installed. Please install it with:")
            print(f"pip install {tool}")

    print("\nChecking optional tools...")
    for tool in optional_tools:
        success, _ = run_command([tool, "--version"])
        results[tool] = success
        if not success:
            print(f"Warning: {tool} is not installed. Some features may be limited.")
            print(f"To install: pip install {tool}")

    return results


def format_project() -> bool:
    """Format all Python files in the project."""
    project_root = Path(__file__).parent.parent
    python_files: list[Any] = []

    # Collect Python files while excluding venv directory
    for file in project_root.glob("**/*.py"):
        # Skip files in venv directory
        if "venv" not in str(file).split("/"):
            python_files.append(str(file))

    if not python_files:
        print("No Python files found to format")
        return True

    # Run black formatter
    print("Running black formatter...")
    if not process_in_batches(
        python_files, ["black", "--line-length", "88", "--target-version", "py37"]
    ):
        return False
    print("Black formatting complete.")

    # Run isort
    print("\nRunning isort to sort imports...")
    if not process_in_batches(
        python_files,
        ["isort", "--profile", "black", "--multi-line", "3", "--line-length", "88"],
    ):
        return False
    print("Import sorting complete.")

    # Run autoflake
    print("\nRunning autoflake to remove unused imports...")
    if not process_in_batches(
        python_files,
        [
            "autoflake",
            "--in-place",
            "--recursive",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--expand-star-imports",
        ],
    ):
        return False
    print("Autoflake cleanup complete.")

    # Try running docformatter if available
    print("\nAttempting to run docformatter...")
    try:
        success = process_in_batches(
            python_files,
            [
                "docformatter",
                "--in-place",
                "--wrap-summaries",
                "88",
                "--wrap-descriptions",
                "88",
            ],
        )
        if success:
            print("Docstring formatting complete.")
        else:
            print("Docformatter encountered some issues - continuing anyway")
    except Exception as e:
        print(f"Docformatter failed - skipping: {str(e)}")

    # Run mypy for type checking
    print("\nRunning mypy type checker...")
    if not process_in_batches(
        python_files, ["mypy", "--ignore-missing-imports", "--strict"]
    ):
        print("Type checking issues found - continuing anyway")
    print("Type checking complete.")

    return True


def setup_pre_commit() -> bool:
    """Set up pre-commit hook configuration."""
    try:
        # First check if we're in a git repository
        success, output = run_command(["git", "rev-parse", "--git-dir"])
        if not success:
            print("Error: Not in a git repository. Please run 'git init' first.")
            return False

        pre_commit_config: str = """
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
        with open(".pre-commit-config.yaml", "w") as f:
            f.write(pre_commit_config)

        print("Installing pre-commit hooks...")
        success, output = run_command(["pre-commit", "install"])
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

    # Check if required tools are available
    required_tools: list[Any] = ["black", "autoflake", "isort", "mypy"]
    missing_required: list[Any] = [tool for tool in required_tools if not dep_results.get(tool)]

    if missing_required:
        print("\nMissing required dependencies:")
        for tool in missing_required:
            print(f"- {tool}")
        print("\nPlease install missing dependencies and try again.")
        return 1

    print("\nStarting code formatting...")
    if not format_project():
        return 1

    print("\nCode formatting completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
