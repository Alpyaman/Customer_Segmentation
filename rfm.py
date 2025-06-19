import pandas as pd
from typing import Optional

def generate_rfm_table(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:

    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    return rfm  # type: ignore