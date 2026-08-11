import pandas_datareader.data as web
from datetime import datetime


def get_fred_series(series_id, start="2020-01-01"):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.today()

    data = web.DataReader(
        series_id,
        "fred",
        start_date,
        end_date,
    )

    return data