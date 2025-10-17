import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str ='1d') -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker symbol.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo'). Default is '1d'.

    Returns:
    pd.DataFrame: DataFrame containing the historical stock data.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return df

def data_ingestion(ticker: str, start_date: str, end_date: str, output_dir: str):
    """
    Ingest stock data and save it to a specified directory.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
    output_dir (str): Directory to save the ingested data.
    """
    df = fetch_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return
    
    output_path = os.path.join(output_dir, f"{ticker}_data.parquet")
    df.to_parquet(output_path)
    print(f"Data for {ticker} saved to {output_path}")


if __name__ == "__main__":
    
    ticker = "SPY"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7*365)
    output_dir = "data"
    
    data_ingestion(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), output_dir)