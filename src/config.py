import yaml
from pathlib import Path
from signals import get_signal
from filters import get_filter
from models import Config


def load_config(config_path: str = "config.yml") -> Config:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config dataclass instance with parsed configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Parse signal
    signal_name = raw_config.get('signal')
    signal = get_signal(signal_name)

    # Parse filters
    filter_names = raw_config.get('filters', [])
    filters = [get_filter(filter_name, signal_name=signal_name) for filter_name in filter_names]

    # Create and return Config dataclass
    return Config(
        name=raw_config.get('name'),
        start=raw_config.get('start'),
        end=raw_config.get('end'),
        rebalance_frequency=raw_config.get('rebalance-frequency'),
        datasets=raw_config.get('datasets', []),
        signal=signal,
        n_bins=raw_config.get('n-bins'),
        weighting_scheme=raw_config.get('weighting-scheme'),
        filters=filters,
        output_path=raw_config.get('output-path'),
        annualize_results=raw_config.get('annualized-results')
    )