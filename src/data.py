import polars as pl
import datetime as dt
import sf_quant.data as sfd
from config import Config

def _get_columns(config: Config) -> list[str]:
    # Get signal columns
    signal_columns = config.signal.columns

    # Get filter columns
    filter_columns = []
    for filter_ in config.filters:
        filter_columns.extend(filter_.columns)

    return list(set(signal_columns + filter_columns))

def _load_universe(dataset: str, start: dt.date, end: dt.date) -> pl.DataFrame:
    match dataset:
        case 'crsp':
            return sfd.load_crsp_daily(
                start=start,
                end=end,
                columns=['date', 'permno']
            )
        
def _get_fwd_return_expr(config: Config) -> pl.Expr:
    holding_period = None
    match config.rebalance_frequency:
        case 'daily':
            holding_period = 1
        case 'monthly':
            holding_period = 21
        case _:
            raise ValueError(f"{config.rebalance_frequency} is not supported!")
        
    return pl.col('return').log1p().rolling_sum(window_size=holding_period).shift(-holding_period).over('permno').alias('fwd_return')

def load_data(config: Config) -> pl.DataFrame:   
    columns = _get_columns(config)

    # Create base dataframe
    data = None
    if 'crsp' in config.datasets:
        data = pl.scan_parquet("data/crsp/crsp_*.parquet")
    
    if data is None:
        raise ValueError("No base dataset provided")
    
    if 'ff3' in config.datasets:
        data = data.join(pl.scan_parquet('data/fama_french/ff3_factors.parquet'), on='date', how='left')

    if 'crsp_ff3_betas' in config.datasets:
        data = (
            data
            .join(
                pl.scan_parquet('data/crsp_ff3_betas/crsp_ff3_betas_*.parquet'),
                on=['date', 'permno'],
                how='left'
            )
        )
        
    data = (
        data
        .select(columns)
        .sort('permno', 'date')
        .with_columns(_get_fwd_return_expr(config=config))
        .filter(
            pl.col('date').is_between(config.start, config.end)
        )
        .sort('permno', 'date')
    )

    return data.collect()