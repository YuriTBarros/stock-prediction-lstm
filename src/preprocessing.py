import pandas as pd 
import os

def preprocess_data(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean and preprocess raw stock data.

    This function do the following steps:
   
    1. Guarantee that the columns are in a unique level (remove multi-index if present).
    2. Add a column for the stock ticker symbol.
    3. Select and reorder relevant columns.
    4. Rename columns to standard names(lowercase).
    5. Convert the 'Date' column to datetime format.
    6. Set the 'Date' column as the DataFrame index and sort by date.

    Parameters:
    raw_df (pd.DataFrame): Raw stock data.
    ticker (str): Stock ticker symbol.

    Returns:
    pd.DataFrame: Preprocessed stock data.
    """
    print("Starting data preprocessing...")

    df = raw_df.copy()

  
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        print("Dropped top-level from MultiIndex columns.")
    
    df.columns.get_level_values(-1)
    df.columns.name = None
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    df['Ticker'] = ticker
    
    df.columns = [col.lower() for col in df.columns]

    return df


def preprocessing_pipeline(raw_data_folder: str, processed_folder_name: str, ticker: str):
    """
    Full preprocessing pipeline: load raw data, preprocess it, and save the processed data.

    Parameters:
    raw_data_folder (str): Path to the folder containing raw data files.
    processed_folder_name (str): Name of the folder to save processed data.
    ticker (str): Stock ticker symbol.
    """
    if not os.path.exists(processed_folder_name):
        os.makedirs(processed_folder_name)
        print(f"Created directory: {processed_folder_name}")

    raw_file_path = os.path.join(raw_data_folder, f"{ticker}_data.parquet")
    processed_file_path = os.path.join(processed_folder_name, f"{ticker}_data_processed.parquet")

    if not os.path.exists(raw_file_path):
        print(f"Raw data file not found: {raw_file_path}")
        return

    raw_df = pd.read_parquet(raw_file_path)
    print(f"Loaded raw data from {raw_file_path}")

    processed_df = preprocess_data(raw_df, ticker)

    processed_df.to_parquet(processed_file_path)
    print(f"Processed data saved to {processed_file_path}")


if __name__ == "__main__":
    raw_data_folder = "data"
    processed_folder_name = "processed_data"
    ticker = "SPY"

    preprocessing_pipeline(raw_data_folder, processed_folder_name, ticker)