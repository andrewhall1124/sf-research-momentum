from pathlib import Path

import yaml

from research.alpha_constructors import get_alpha_constructor
from research.constraints import get_constraint
from research.filters import get_filter
from research.models import MVEBacktestConfig, QuantileBacktestConfig
from research.signals import get_signal


def load_quantile_backtest_config(
    config_path: str = "quantile_backtest_cfg.yml",
) -> QuantileBacktestConfig:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        QuantileBacktestConfig dataclass instance with parsed configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Parse datasets
    datasets = raw_config.get("datasets", [])

    # Set id_col
    id_col = None
    if "crsp" in datasets:
        id_col = "permno"
    elif "barra" in datasets:
        id_col = "barrid"
    else:
        raise RuntimeError("No base dataset included!")

    # Parse signal
    signal_name = raw_config.get("signal")
    signal = get_signal(signal_name, id_col=id_col)

    # Parse filters
    filter_names = raw_config.get("filters", [])
    filters = [
        get_filter(filter_name, signal_name=signal_name) for filter_name in filter_names
    ]

    return QuantileBacktestConfig(
        name=raw_config.get("name"),
        start=raw_config.get("start"),
        end=raw_config.get("end"),
        rebalance_frequency=raw_config.get("rebalance-frequency"),
        datasets=datasets,
        signal=signal,
        n_bins=raw_config.get("n-bins"),
        weighting_scheme=raw_config.get("weighting-scheme"),
        filters=filters,
        output_path=raw_config.get("output-path"),
        annualize_results=raw_config.get("annualized-results"),
    )


def load_mve_backtest_config(
    config_path: str = "mve_backtest_cfg.yml",
) -> MVEBacktestConfig:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        MVEBacktestConfig dataclass instance with parsed configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Parse datasets
    datasets = raw_config.get("datasets", [])

    # Set id_col
    id_col = None
    if "crsp" in datasets:
        id_col = "permno"
    elif "barra" in datasets:
        id_col = "barrid"
    else:
        raise RuntimeError("No base dataset included!")

    # Parse signal
    signal_name = raw_config.get("signal")
    signal = get_signal(signal_name, id_col=id_col)

    # Parse filters
    filter_names = raw_config.get("filters", [])
    filters = [
        get_filter(filter_name, signal_name=signal_name) for filter_name in filter_names
    ]

    # Parse constraints
    constraint_names = raw_config.get("constraints", [])
    constraints = [
        get_constraint(constraint_name) for constraint_name in constraint_names
    ]

    # Parse alpha constructor
    alpha_constructor_name = raw_config.get("alpha-constructor")
    alpha_constructor = get_alpha_constructor(
        alpha_constructor_name, signal_name=signal_name
    )

    return MVEBacktestConfig(
        name=raw_config.get("name"),
        start=raw_config.get("start"),
        end=raw_config.get("end"),
        rebalance_frequency=raw_config.get("rebalance-frequency"),
        datasets=datasets,
        signal=signal,
        gamma=raw_config.get("gamma"),
        filters=filters,
        constraints=constraints,
        output_path=raw_config.get("output-path"),
        annualize_results=raw_config.get("annualized-results"),
        alpha_constructor=alpha_constructor,
    )
