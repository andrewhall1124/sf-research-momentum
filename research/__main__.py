#!/usr/bin/env python3
"""CLI for running momentum backtests."""

import click
from pathlib import Path
from config import load_config
from backtest import backtest


@click.group()
def cli():
    """Momentum research backtesting CLI."""
    pass


@cli.command()
@click.option("--config-path", type=click.Path(exists=True, path_type=Path), default="config.yml")
def run(config_path: Path):
    """
    Run a backtest using the specified config file.

    CONFIG_PATH: Path to the YAML configuration file

    Example:
        python -m research run config.yml
        python -m research run configs/mom/mom-in-sample-equal.yml
    """
    click.echo(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    click.echo(f"Starting backtest: {config.name}")
    backtest(config)

    click.echo(f"Backtest completed! Results saved to: {config.output_path}")


@cli.command()
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="configs",
    help="Directory containing config subdirectories (default: configs)",
)
def run_all(config_dir: Path):
    """
    Run backtests for all config files found in the config directory.

    This command recursively searches for all .yml files in the config directory
    and runs a backtest for each one.

    Example:
        python -m research run-all
        python -m research run-all --config-dir configs
    """
    # Find all .yml files recursively
    config_files = sorted(config_dir.rglob("*.yml"))

    if not config_files:
        click.echo(f"No config files found in {config_dir}", err=True)
        return

    click.echo(f"Found {len(config_files)} config file(s) in {config_dir}")
    click.echo()

    for i, config_path in enumerate(config_files, 1):
        click.echo(f"[{i}/{len(config_files)}] Processing: {config_path.relative_to(config_dir.parent)}")

        try:
            config = load_config(str(config_path))
            click.echo(f"  Running: {config.name}")
            backtest(config)
            click.echo(f"   Completed! Results saved to: {config.output_path}")
        except Exception as e:
            click.echo(f"   Error: {e}", err=True)
            continue

        click.echo()

    click.echo(f"All backtests completed! Processed {len(config_files)} config(s).")


if __name__ == "__main__":
    cli()
