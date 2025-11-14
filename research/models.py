import datetime as dt
from dataclasses import dataclass

import polars as pl
import sf_quant.optimizer.constraints


@dataclass
class Signal:
    name: str
    expr: pl.Expr
    columns: list[str]
    lookback_days: int


@dataclass
class AlphaConstructor:
    name: str
    expr: pl.Expr
    columns: list[str]


@dataclass
class Filter:
    name: str
    expr: pl.Expr
    columns: list[str]


@dataclass
class Constraint:
    name: str
    constraint: sf_quant.optimizer.constraints.Constraint
    columns: list[str]


@dataclass
class Dataset:
    name: str
    primary_keys: list[str]
    source: str


@dataclass
class QuantileBacktestConfig:
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


@dataclass
class MVEBacktestConfig:
    name: str
    start: dt.date
    end: dt.date
    rebalance_frequency: str
    datasets: list[str]
    signal: Signal
    alpha_constructor: AlphaConstructor
    gamma: float
    filters: list[Filter]
    constraints: list[Constraint]
    output_path: str
    annualize_results: bool
