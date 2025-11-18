import datetime as dt
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
from tqdm import tqdm


def crsp_history_flow(start: dt.date, end: dt.date) -> None:
    data = (
        sfd.load_crsp_daily(
            start=start,
            end=end,
            columns=["date", "permno", "ticker", "prc", "ret", "shrout"],
        )
        .rename({"prc": "price", "ret": "return", "shrout": "shares"})
        .with_columns(pl.col("shares").mul(pl.col("price")).alias("market_cap"))
    )

    years = list(range(start.year, end.year + 1))

    for year in tqdm(years, "Loading CRSP daily data"):
        file_path = Path(f"data/crsp/crsp_{year}.parquet")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        year_start = dt.date(year, 1, 1) if year > start.year else start
        year_end = dt.date(year, 12, 31) if year < end.year else end

        year_data = data.filter(pl.col("date").is_between(year_start, year_end))

        year_data.write_parquet(file_path)


if __name__ == "__main__":
    crsp_history_flow(start=dt.date(1925, 1, 1), end=dt.date.today())
