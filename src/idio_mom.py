import polars as pl
import sf_quant.data as sfd
import datetime as dt

start = dt.date(1963, 7, 31)
end = dt.date(2015, 12, 31)

df = (
    sfd.load_crsp_daily(
        start=start,
        end=end,
        columns=[
            'date',
            'permno',
            'ret',
            'prc',
            'shrout'
        ]
    )
)

print(df)