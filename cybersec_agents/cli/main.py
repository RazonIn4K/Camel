"""Command Line Interface for Cybersecurity Agents."""

import argparse
import sys
from typing import List


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cybersecurity Agents CLI")
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("--output", help="Output file path")
    return parser.parse_args(args)


def main() -> int:
    """Main CLI entry point."""
    args = parse_args(sys.argv[1:])
    # Add CLI logic here
    return 0


if __name__ == "__main__":
    sys.exit(main())
