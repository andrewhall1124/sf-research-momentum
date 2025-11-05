from src.models import Filter, Config
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

def null_signal(signal_name: str) -> Filter:
    return Filter(
        name=null_signal.__name__,
        expr=pl.col(signal_name).is_not_null(),
        columns=[]
    )

def get_filter(name: str, **kwargs) -> Filter:
    signal_name = kwargs.get('signal_name')
    match name:
        case 'penny-stocks':
            return penny_stocks()
        case 'micro-caps':
            return micro_caps()
        case 'null-signal':
            return null_signal(signal_name)
        case _:
            raise ValueError
        
def apply_filters(signals: pl.DataFrame, config: Config) -> pl.DataFrame:
    return (
        signals
        .filter(
            [filter_.expr for filter_ in config.filters]
        )
        .sort('permno', 'date')
    )