from pathlib import Path
import subprocess
import datetime as dt
import polars as pl

import sf_quant.backtester as sfb

from research.alpha_constructors import construct_alphas
from research.data import load_data
from research.evaluations import (
    create_mve_returns_chart,
    create_mve_summary_table,
    create_quantile_returns_chart,
    create_quantile_summary_table,
)
from research.filters import apply_filters
from research.models import MVEBacktestConfig, QuantileBacktestConfig, Constraint
from research.portfolios import construct_mve_portfolios, construct_quantile_portfolios
from research.returns import construct_returns, construct_returns_from_weights
from research.signals import construct_signals


def quantile_backtest(config: QuantileBacktestConfig):
    print("Loading data...")
    data = load_data(
        start=config.start,
        end=config.end,
        rebalance_frequency=config.rebalance_frequency,
        datasets=config.datasets,
        signal=config.signal,
        filters=config.filters,
    )

    print("Constructing signals...")
    signals = construct_signals(data=data, signal=config.signal)

    print("Applying filters...")
    filtered = apply_filters(signals=signals, filters=config.filters)

    print("Constructing portfolios...")
    portfolios = construct_quantile_portfolios(filtered, config=config)

    print("Constructing returns...")
    returns = construct_returns(portfolios, config=config)

    print("Saving results...")
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_quantile_summary_table(returns=returns, config=config, file_path=output_path)
    create_quantile_returns_chart(returns=returns, config=config, file_path=output_path)


def mve_backtest(config: MVEBacktestConfig):
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(
        start=config.start,
        end=config.end,
        rebalance_frequency=config.rebalance_frequency,
        datasets=config.datasets,
        signal=config.signal,
        filters=config.filters,
        constraints=config.constraints,
        alpha_constructor=config.alpha_constructor,
    )

    print("Constructing signals...")
    signals = construct_signals(data=data, signal=config.signal)

    print("Applying filters...")
    filtered = apply_filters(signals=signals, filters=config.filters)

    print("Constructing alphas...")
    alphas = construct_alphas(
        signals=filtered, alpha_constructor=config.alpha_constructor
    )

    print("Constructing portfolios...")
    weights = construct_mve_portfolios(
        alphas=alphas,
        rebalance_frequency=config.rebalance_frequency,
        gamma=config.gamma,
        constraints=config.constraints,
    )

    print("Saving weights...")
    weights.write_parquet(output_path.with_suffix(".parquet"))

    print("Constructing returns...")
    returns = construct_returns_from_weights(
        weights=weights, rebalance_frequency=config.rebalance_frequency
    )

    print("Saving results...")
    create_mve_summary_table(
        returns=returns,
        file_path=output_path,
        annualize_results=config.annualize_results,
        rebalance_frequency=config.rebalance_frequency,
        name=config.name,
        start=config.start,
        end=config.end,
    )
    create_mve_returns_chart(returns=returns, name=config.name, file_path=output_path)


def _submit_year_job(
    signal_name: str,
    year: int,
    dataset_path: str = "alphas.parquet",
    config_path: str = "mve_backtest_cfg.yml",
    n_cpus: int = 8,
):
    """Submit a single slurm job for one signal/year combination."""

    output_path = f"weights/{signal_name}/{signal_name}_{year}.parquet"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=backtest_{signal_name}_{year}
#SBATCH --output=logs/{signal_name}/out/backtest_{signal_name}_{year}.out
#SBATCH --error=logs/{signal_name}/err/backtest_{signal_name}_{year}.err
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --mail-user=amh1124@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source .venv/bin/activate
echo "Running signal={signal_name} for {year}"
srun python -m research run-single-year-mve-backtest \\
    --config-path "{config_path}" \\
    --signal-name "{signal_name}" \\
    --year {year} \\
    --alphas-path "{dataset_path}" \\
    --output-path "{output_path}"
"""

    result = subprocess.run(
        ["sbatch"], input=sbatch_script, capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"Submitted {signal_name} {year}: {result.stdout.strip()}")
    else:
        print(f"Error: {result.stderr}")

    return result


def mve_backtest_parallel(config: MVEBacktestConfig):
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(
        start=config.start,
        end=config.end,
        rebalance_frequency=config.rebalance_frequency,
        datasets=config.datasets,
        signal=config.signal,
        filters=config.filters,
        constraints=config.constraints,
        alpha_constructor=config.alpha_constructor,
    )

    print("Constructing signals...")
    signals = construct_signals(data=data, signal=config.signal)

    print("Applying filters...")
    filtered = apply_filters(signals=signals, filters=config.filters)

    print("Constructing alphas...")
    alphas = construct_alphas(
        signals=filtered, alpha_constructor=config.alpha_constructor
    )

    print("Saving alphas temporarily...")
    alphas.write_parquet("alphas.parquet")

    print("Constructing portfolios...")
    years = alphas['date'].dt.year().unique().sort().to_list()

    for year in years:
        _submit_year_job(signal_name=config.signal.name, year=year)


def single_year_backtest(
    gamma: float,
    year: int,
    alphas_path: Path,
    constraints: list[Constraint],
    rebalance_frequency: str
):
    alphas = (
        pl.scan_parquet(alphas_path)
        .filter(pl.col("date").dt.year().eq(year))
        .select("date", "barrid", "alpha", "predicted_beta")
        .collect()
    )

    if alphas.is_empty():
        print("[WARNING] After filtering, input df was empty.")
        return None

    return construct_mve_portfolios(alphas=alphas, rebalance_frequency=rebalance_frequency, constraints=constraints, gamma=gamma)
