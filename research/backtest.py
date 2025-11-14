from pathlib import Path

from research.alpha_constructors import construct_alphas
from research.data import load_data
from research.evaluations import (create_mve_returns_chart,
                                  create_mve_summary_table,
                                  create_quantile_returns_chart,
                                  create_quantile_summary_table)
from research.filters import apply_filters
from research.models import MVEBacktestConfig, QuantileBacktestConfig
from research.portfolios import (construct_mve_portfolios,
                                 construct_quantile_portfolios)
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
        constraints=None,
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
