import datetime as dt
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
from tqdm import tqdm


def barra_history_flow(start: dt.date, end: dt.date) -> None:
    data = sfd.load_assets(
        start=start,
        end=end,
        columns=[
            "date",
            "barrid",
            "ticker",
            "price",
            "return",
            "predicted_beta",
            "specific_risk",
            "market_cap",
        ],
        in_universe=True,
    ).with_columns(pl.col("return", "specific_risk").truediv(100))

    years = list(range(start.year, end.year + 1))

    for year in tqdm(years, "Loading Barra daily data"):
        file_path = Path(f"data/barra/barra_{year}.parquet")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        year_start = dt.date(year, 1, 1) if year > start.year else start
        year_end = dt.date(year, 12, 31) if year < end.year else end

        year_data = data.filter(pl.col("date").is_between(year_start, year_end))

        year_data.write_parquet(file_path)


if __name__ == "__main__":
    barra_history_flow(start=dt.date(1995, 7, 31), end=dt.date.today())
