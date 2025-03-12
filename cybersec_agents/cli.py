from typing import Optional

import click

from .core.service_wrapper import CyberSecurityService


@click.group()
def cli():
    """Cybersecurity Agents CLI."""


@cli.command()
@click.option("--provider", default="anthropic", help="AI provider to use")
@click.option("--config", type=str, help="Path to config file")
def analyze(provider: str, config: Optional[str] = None):
    """Run security analysis."""
    service = CyberSecurityService(provider=provider, config_path=config)
    service.process_command("analyze")  # Use the service instance


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
