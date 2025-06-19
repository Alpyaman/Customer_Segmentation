import pandas as pd

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(file_path, encoding='latin-1')

    df = df.dropna(subset=['CustomerID']).copy()

    df = df[~df['InvoiceNo'].str.contains('C', na=False)].copy()

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    return df  # type: ignore