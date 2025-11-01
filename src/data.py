import sf_quant.data as sfd
import datetime as dt
import polars as pl

start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)

columns = [
    'date',
    'barrid',
    'price',
    'return',
]

assets = (
    sfd.load_assets(
        start=start,
        end=end,
        columns=columns,
        in_universe=True
    )
    .with_columns(
        pl.col('return').truediv(100)
    )
)

def cross_sectional_momentum() -> pl.Expr:
    return pl.col('return').log1p().rolling_sum(window_size=230).shift(22).over('barrid').alias(cross_sectional_momentum.__name__)

def time_series_momentum() -> pl.Expr:
    return pl.col('return')

signals = (
    assets
    .sort('barrid', 'date')
    .with_columns(
        cross_sectional_momentum()
    )
    .drop_nulls('cross_sectional_momentum')
)

print(signals)

