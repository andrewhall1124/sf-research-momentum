import io
import urllib.request
import zipfile
from pathlib import Path

import polars as pl


def momentum_factor_returns_flow() -> None:
    # URL for the Fama-French 3 factors (monthly data)
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/6_Portfolios_ME_Prior_12_2_Daily_CSV.zip"

    # Download the zip file
    urllib.request.urlretrieve(mom_url, "fama_french.zip")

    # Extract the CSV file
    with zipfile.ZipFile("fama_french.zip", "r") as zip_file:
        zip_file.extractall()

    # Read the CSV file with polars
    # Skip first 4 rows (header info) and read only the monthly data
    # The file has annual data after the monthly data, which we'll exclude
    with open("6_Portfolios_ME_Prior_12_2_Daily.csv", "r") as f:
        lines = f.readlines()

    # Find where monthly data ends (when we hit empty line or "Annual" section)
    value_weight_end_idx = None
    for i, line in enumerate(lines[11:], start=11):  # Start after header rows
        if line.strip() == "" in line:
            print(line)
            value_weight_end_idx = i
            break

    # Read only the value weight data section
    csv_data = "".join(lines[11:value_weight_end_idx])

    # Parse with polars
    data = pl.read_csv(
        io.StringIO(csv_data),
        has_header=True,
        new_columns=["date", "sl", "sn", "sw","bl","bn","bw"]
    ).with_columns(
        [
            # Convert date format (YYYYMM) to actual date (end of month)
            pl.col("date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d"),
            # Convert from percentage to decimal
            pl.col("sl").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("sn").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("sw").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("bl").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("bn").str.strip_chars().cast(pl.Float64).truediv(100),
            pl.col("bw").str.strip_chars().cast(pl.Float64).truediv(100),
        ]
    ).with_columns(
        # MOM = (BW+SW) / 2 - (BL + SL) / 2
        pl.sum_horizontal('bw', 'sw').truediv(2).sub(
            pl.sum_horizontal('bl', 'sl').truediv(2)
        ).alias('mom')
    )

    # Create output directory
    file_path = Path("data/momentum_factor_returns/momentum_factor_returns.parquet")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    data.write_parquet(file_path)

    # Clean up downloaded files
    Path("fama_french.zip").unlink(missing_ok=True)
    Path("6_Portfolios_ME_Prior_12_2_Daily.csv").unlink(missing_ok=True)

if __name__ == '__main__':
    momentum_factor_returns_flow()