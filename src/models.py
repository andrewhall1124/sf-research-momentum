from dataclasses import dataclass
import polars as pl

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