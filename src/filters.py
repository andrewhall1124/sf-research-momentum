from src.models import Filter
import polars as pl

def penny_stocks() -> Filter:
    return Filter(
        name=penny_stocks.__name__,
        expr=pl.col('price').gt(1),
        columns=['price']
    )

def micro_caps() -> Filter:
    return Filter(
        name=micro_caps.__name__,
        expr=pl.col('market_cap').gt(pl.col('market_cap').quantile(.20).over('date')),
        columns=['date', 'market_cap']
    )

def null_signal() -> Filter:
    return Filter(
        name=null_signal.__name__,
        expr=pl.col('signal').is_not_null(),
        columns=['signal']
    )

def get_filter(name: str) -> Filter:
    match name:
        case 'penny-stocks':
            return penny_stocks()
        case 'micro-caps':
            return micro_caps()
        case 'null-signal':
            return null_signal()
        case _:
            raise ValueError