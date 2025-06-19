import pandas as pd

def load_and_clean_data(data):
    # If path provided, load CSV
    if isinstance(data, str):
        df = pd.read_csv(data, encoding="latin-1")
    else:
        df = data.copy()

    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df
