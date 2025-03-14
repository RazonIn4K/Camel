from typing import Optional

import click

from .forensics_planner import ForensicsPlanner
from .network_anomaly_detector import NetworkAnomalyDetector


@click.group()
def cli():
    """Camel AI Cybersecurity CLI."""


@cli.command()
@click.argument("command")
@click.option("--output", "-o", help="Output file path")
@click.option(
    "--format", "-f", default="text", type=click.Choice(["text", "json", "yaml"])
)
def run(command: str, output: Optional[str], format: str):
    """Execute a cybersecurity analysis command."""
    if command.startswith("analyze"):
        detector = NetworkAnomalyDetector()
        result = detector.analyze_command(command)
        _handle_output(result, output, format)
    elif command.startswith("plan"):
        planner = ForensicsPlanner()
        result = planner.create_plan(command)
        _handle_output(result, output, format)


def _handle_output(result: dict, output: Optional[str], format: str):
    """Handle command output in specified format."""
    formatted_result = _format_result(result, format)
    if output:
        with open(output, "w") as f:
            f.write(formatted_result)
    else:
        click.echo(formatted_result)


if __name__ == "__main__":
    cli()
