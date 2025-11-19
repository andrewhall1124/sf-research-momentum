import polars as pl
import datetime as dt

start = dt.date(1930, 1, 1)
end = dt.date(2017, 12, 31)

month_end_dates = (
    pl.scan_parquet('data/momentum_factor_returns/momentum_factor_returns.parquet')
    .with_columns(
        pl.col('date').dt.strftime("%Y%m").alias('year_month')
    )
    .group_by('year_month')
    .agg(
        pl.col('date').max()
    )
    .select('date')
)

mom = (
    pl.scan_parquet('data/momentum_factor_returns/momentum_factor_returns.parquet')
    .sort('date')
    .select(
        'date',
        pl.col('mom').alias('return'),
        pl.col('mom').log1p().rolling_sum(window_size=21).exp().sub(1).alias('mom_monthly')
    )
    .filter(pl.col('date').is_between(start, end))
    .sort('date')
    .collect()
)

vol_target = 4.11 / 100
cmom = (
    mom
    .with_columns(
        pl.col('mom').pow(2).rolling_sum(126).mul(21/126).alias('vol_forecast')
    )
    .with_columns(
        pl.lit(vol_target).truediv('vol_forecast').alias('weight')
    )
    .with_columns(
        pl.col('mom_monthly').mul('weight')
    )
)

print(cmom)

# mom_monthly = (
#     mom
#     .join(
#         other=month_end_dates,
#         on='date',
#         how='inner'
#     )
# )

# print(mom)

# annual_factor = 12
# summary_table = (
#     mom
#     .select(
#         pl.col('mom_monthly').mean().mul(100).alias('mean_return'),
#         pl.col('mom_monthly').std().mul(100).alias('volatility')
#     )
#     .with_columns(
#         pl.col('mean_return').truediv('volatility').mul(pl.lit(annual_factor).sqrt()).alias('sharpe')
#     )
#     .with_columns(
#         pl.all().round(2)
#     )
# )

# print(summary_table)

