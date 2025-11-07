from config import load_config
from data import load_data
from signals import construct_signals
from filters import apply_filters
from portfolios import construct_portfolios
from returns import construct_returns
from evaluations import create_summary_table, create_quantile_returns_chart
from models import Config
from pathlib import Path


def backtest(config: Config):
    print("Loading data...")
    data = load_data(config)

    print("Constructing signals...")
    signals = construct_signals(data=data, config=config)

    print("Applying filters...")
    filtered = apply_filters(signals=signals, config=config)

    print("Constructing portfolios...")
    portfolios = construct_portfolios(filtered, config=config)

    print("Constructing returns...")
    returns = construct_returns(portfolios, config=config)

    print("Saving results...")
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_summary_table(returns=returns, config=config, file_path=output_path)
    create_quantile_returns_chart(returns=returns, config=config, file_path=output_path)

