from dataclasses import dataclass
import polars as pl
import datetime as dt

@dataclass
class Signal:
    name: str
    expr: pl.Expr
    columns: list[str]
    lookback_days: int

@dataclass
class Filter:
    name: str
    expr: pl.Expr
    columns: list[str]

@dataclass
class Dataset:
    name: str
    primary_keys: list[str]
    source: str

@dataclass
class Config:
    """Configuration dataclass for backtest parameters."""

    name: str
    start: dt.date
    end: dt.date
    rebalance_frequency: str
    datasets: list[str]
    signal: Signal
    n_bins: int
    weighting_scheme: str
    filters: list[Filter]
    output_path: str
    annualize_results: bool