import pandas as pd


def load_data(file_path):
    """
    Load dataset from CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise