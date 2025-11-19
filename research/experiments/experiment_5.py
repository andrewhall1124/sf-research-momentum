import polars as pl
import datetime as dt
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm

start = dt.date(1930, 1, 1)
end = dt.date(2017, 12, 31)

# Load data and get month-end dates
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
    .collect()
)

# Get daily momentum returns
daily_returns = (
    pl.scan_parquet('data/momentum_factor_returns/momentum_factor_returns.parquet')
    .filter(pl.col('date').is_between(start, end))
    .sort('date')
    .select(
        'date',
        pl.col('mom').alias('return')
    )
    .collect()
)

# Calculate monthly returns at month-ends
mom_monthly = (
    daily_returns
    .with_columns(
        # Calculate cumulative return over past ~21 days (month)
        pl.col('return').log1p()
        .rolling_sum(window_size=21)
        .exp().sub(1)
        .shift(-21)
        .alias('return_monthly')
    )
    .with_columns(
        pl.lit('mom').alias('signal')
    )
    .select('date', 'signal', 'return_monthly')
    .join(month_end_dates, on='date', how='inner')
    .sort('date')
)

# Calculate volatility-scaled version
vol_target = 4.35 / 100 # Monthly volatility target

cmom_monthly = (
    daily_returns
    .sort('date')
    .with_columns(
        # Calculate realized variance using daily squared returns
        # 126 trading days ≈ 6 months
        pl.col('return').pow(2)
        .truediv(126)
        .rolling_sum(window_size=126)
        .mul(21) # Scale to monthly
        .sqrt()
        .alias('vol_forecast')
    )
    .with_columns(
        # Calculate weight based on volatility (shifted to avoid look-ahead)
        pl.lit(vol_target).truediv(pl.col('vol_forecast')).alias('weight')
    )
    .with_columns(
        # Apply weight to calculate scaled returns
        pl.col('return').log1p()
        .rolling_sum(window_size=21)
        .exp().sub(1)
        .shift(-21)
        .alias('return_monthly')
    )
    .with_columns(
        pl.col('return_monthly').mul(pl.col('weight'))
    )
    .with_columns(
        pl.lit('cmom').alias('signal')
    )
    .select('date', 'signal', 'return_monthly')
    .join(month_end_dates, on='date', how='inner')
    .sort('date')
)

# Calculate semi-variance scaled version (SMOM)
smom_monthly = (
    daily_returns
    .with_columns(
        # Create indicator for negative returns, then square only negative returns
        pl.when(pl.col('return') < 0)
        .then(pl.col('return').pow(2))
        .otherwise(0)
        .alias('return_squared_neg')
    )
    .with_columns(
        # Semi-variance: sum of squared negative returns over 126 days
        pl.col('return_squared_neg')
        .rolling_sum(window_size=126)
        .mul(21/126)  # Scale to monthly
        .sqrt()
        .alias('semi_vol_forecast')
    )
    .with_columns(
        pl.lit(vol_target).truediv(pl.col('semi_vol_forecast')).alias('weight')
    )
    .with_columns(
        pl.col('return').log1p()
        .rolling_sum(window_size=21)
        .exp().sub(1)
        .shift(-12)
        .alias('return_monthly')
    )
    .with_columns(
        pl.col('return_monthly').mul(pl.col('weight'))
    )
    .with_columns(
        pl.lit('smom').alias('signal')
    )
    .select('date', 'signal', 'return_monthly')
    .join(month_end_dates, on='date', how='inner')
    .sort('date')
)

market_data = (
    pl.scan_parquet("data/fama_french_factors/fama_french_factors.parquet")
    .join(
        pl.scan_parquet('data/momentum_factor_returns/momentum_factor_returns.parquet')
        .select('date', pl.col('mom').alias('r_mom')),
        on='date',
        how='left'
    )
    .select(
        'date',
        'r_mom',
        pl.col('mkt_rf').add(pl.col('rf')).alias('rm'),
        'rf',
        pl.col('mkt_rf').alias('rmrf')
    )
    .filter(
        pl.col('date').is_between(start, end)
    )
    .sort('date')
    .with_columns(
        pl.col('rm').log1p().rolling_sum(252 * 2).exp().sub(1).alias('rm_2y')
    )
    .with_columns(
        pl.when(pl.col('rm_2y').lt(0)).then(pl.lit(1))
        .when(pl.col('rm_2y').ge(0)).then(pl.lit(0))
        .alias('bear_indicator')
    )
    .with_columns(
        pl.col('rmrf').rolling_var(126).alias('rmrf_variance')
    )
    .with_columns(
        pl.col('bear_indicator').mul(pl.col('rmrf_variance')).alias('interaction')
    )
    .select('date', 'r_mom', 'interaction')
    .collect()
)

month_dates = (
    market_data
    .with_columns(
        pl.col('date').dt.strftime('%Y%m').alias('year_month')
    )
    .group_by('year_month')
    .agg(
        pl.col('date').max()
    )
    ['date']
    .unique()
    .sort()
    .to_list()
)

def estimate_coefficients(data: pl.DataFrame) -> dict[str, None | float]:
    data = data.drop_nulls()

    if not len(data) > 0:
        return {
        'gamma_0': None,
        'gamma_1': None        
    }
    
    formula = "r_mom ~ interaction"
    model = smf.ols(formula=formula, data=data)
    results = model.fit()
    
    gamma_0 = results.params['Intercept']
    gamma_1 = results.params['interaction']
    
    return {
        'gamma_0': gamma_0,
        'gamma_1': gamma_1
    }

coefficients_list = []
for month_date in tqdm(month_dates, "Computing coefficients"):
    regression_data_subset = market_data.filter(pl.col('date').le(month_date))
    coefficients = estimate_coefficients(data=regression_data_subset)
    coefficients_list.append({'date': month_date} | coefficients)

coefficients = pl.DataFrame(coefficients_list)

lambda_ = 1.5e-2
# Calculate monthly returns at month-ends
dmom_monthly = (
    daily_returns
    .join(coefficients, on='date', how='left')
    .join(market_data, on='date', how='left')
    .sort('date')
    .with_columns(
        # Calculate realized variance using daily squared returns
        # 126 trading days ≈ 6 months
        pl.col('return').pow(2)
        .truediv(126)
        .rolling_sum(window_size=126)
        .mul(21) # Scale to monthly
        .sqrt()
        .alias('vol_forecast')
    )
    .with_columns(
        # Calculate weight based on volatility (shifted to avoid look-ahead)
        pl.lit(vol_target).truediv(pl.col('vol_forecast')).alias('weight')
    )
    .with_columns(
        pl.col('gamma_0').add(pl.col('gamma_1').mul('interaction')).alias('return_forecast')
    )
    .with_columns(
        pl.col('return_forecast').truediv(pl.col('vol_forecast')).mul(1/(2*lambda_)).alias('weight')
    )
    .with_columns(
        # Calculate cumulative return over past ~21 days (month)
        pl.col('return').log1p()
        .rolling_sum(window_size=21)
        .exp().sub(1)
        .shift(-21)
        .alias('return_monthly')
    )
    .with_columns(
        pl.col('return_monthly').mul('weight')
    )
    .with_columns(
        pl.lit('dmom').alias('signal')
    )
    .select('date', 'signal', 'return_monthly')
    .join(month_end_dates, on='date', how='inner')
    .sort('date')
)


# Combine and calculate summary statistics
returns_monthly = (
    pl.concat([
        mom_monthly,
        cmom_monthly,
        smom_monthly,
        dmom_monthly
    ])
    .sort('date')
)

annual_factor = 12
summary_table = (
    returns_monthly
    .group_by('signal')
    .agg(
        pl.col('return_monthly').mean().mul(100).alias('mean_return'),
        pl.col('return_monthly').std().mul(100).alias('volatility')
    )
    .with_columns(
        pl.col('mean_return').truediv(pl.col('volatility')).mul(pl.lit(annual_factor).sqrt()).alias('sharpe')
    )
    .with_columns(
        pl.exclude('signal').round(2)
    )
    .sort('signal')
)

print(summary_table)