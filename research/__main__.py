#!/usr/bin/env python3
"""CLI for running momentum backtests."""

from pathlib import Path

import click

from research.backtest import mve_backtest, mve_backtest_parallel, quantile_backtest, single_year_backtest
from research.config import load_mve_backtest_config, load_quantile_backtest_config


@click.group()
def cli():
    """Momentum research backtesting CLI."""
    pass


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, path_type=Path),
    default="quantile_backtest_cfg.yml",
)
def run_quantile_backtest(config_path: Path):
    """
    Run a quantile backtest using the specified config file.

    CONFIG_PATH: Path to the YAML configuration file

    Example:
        python -m research run quantile_backtest_cfg.yml
        python -m research run configs/quantile/mom/mom-in-sample-equal.yml
    """
    click.echo(f"Loading config from: {config_path}")
    config = load_quantile_backtest_config(str(config_path))

    click.echo(f"Starting backtest: {config.name}")
    quantile_backtest(config)

    click.echo(f"Backtest completed! Results saved to: {config.output_path}")


@cli.command()
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="configs/quantile",
    help="Directory containing config subdirectories (default: configs/quantile)",
)
def run_all_quantile_backtest(config_dir: Path):
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
        click.echo(
            f"[{i}/{len(config_files)}] Processing: {config_path.relative_to(config_dir.parent)}"
        )

        try:
            config = load_quantile_backtest_config(str(config_path))
            click.echo(f"  Running: {config.name}")
            quantile_backtest(config)
            click.echo(f"   Completed! Results saved to: {config.output_path}")
        except Exception as e:
            click.echo(f"   Error: {e}", err=True)
            continue

        click.echo()

    click.echo(f"All backtests completed! Processed {len(config_files)} config(s).")


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, path_type=Path),
    default="mve_backtest_cfg.yml",
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Run backtest in parallel using SLURM jobs for each year",
)
def run_mve_backtest(config_path: Path, parallel: bool):
    """
    Run a MVE backtest using the specified config file.

    CONFIG_PATH: Path to the YAML configuration file

    Example:
        python -m research run mve_backtest_cfg.yml
        python -m research run configs/quantile/mom/mom-select-sample-zero-beta.yml
        python -m research run mve_backtest_cfg.yml --parallel
    """
    click.echo(f"Loading config from: {config_path}")
    config = load_mve_backtest_config(str(config_path))

    click.echo(f"Starting backtest: {config.name}")
    if parallel:
        click.echo("Running in parallel mode (submitting SLURM jobs)...")
        mve_backtest_parallel(config)
        click.echo("SLURM jobs submitted! Check logs/ directory for job outputs.")
    else:
        mve_backtest(config)
        click.echo(f"Backtest completed! Results saved to: {config.output_path}")


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, path_type=Path),
    default="mve_backtest_cfg.yml",
)
@click.option(
    "--signal-name",
    type=str,
    required=True,
    help="Name of the signal to backtest",
)
@click.option(
    "--year",
    type=int,
    required=True,
    help="Year to run the backtest for",
)
@click.option(
    "--alphas-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to alphas",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save the backtest results",
)
def run_single_year_mve_backtest(
    config_path: Path,
    signal_name: str,
    year: int,
    alphas_path: Path,
    output_path: Path,
):
    config = load_mve_backtest_config(config_path)
    constraints = [c.constraint for c in config.constraints]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = single_year_backtest(
        gamma=config.gamma,
        year=year,
        alphas_path=alphas_path,
        constraints=constraints,
        rebalance_frequency=config.rebalance_frequency
    )

    if weights is not None:
        weights.write_parquet(output_path.with_suffix(".parquet"))
    else:
        click.echo(f"No weights generated for {signal_name} {year} - alphas may be empty")


if __name__ == "__main__":
    cli()
