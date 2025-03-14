#!/usr/bin/env python3
"""Script to format code and push changes to GitHub."""

import subprocess
import sys
from typing import List, Tuple


def run_command(command: List[str]) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result: Any = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def push_to_github(commit_msg: str, no_verify: bool = False) -> bool:
    """Push changes to GitHub with the given commit message."""
    try:
        # Add all changes
        success, output = run_command(["git", "add", "."])
        if not success:
            print(f"Error adding files: {output}")
            return False

        # Commit changes
        commit_cmd: list[Any] = ["git", "commit", "-m", commit_msg]
        if no_verify:
            commit_cmd.append("--no-verify")

        success, output = run_command(commit_cmd)
        if not success:
            print(f"Error committing changes: {output}")
            return False

        # Push to GitHub
        success, output = run_command(["git", "push", "origin", "main"])
        if not success:
            print(f"Error pushing to GitHub: {output}")
            return False

        return True

    except Exception as e:
        print(f"Error during git operations: {str(e)}")
        return False


def main() -> int:
    """Main function."""
    try:
        commit_msg = sys.argv[1] if len(sys.argv) > 1 else "Update"
        success = push_to_github(commit_msg)

        if not success:
            print("Failed to push changes to GitHub")
            return 1

        print("Successfully pushed changes to GitHub")
        return 0

    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
