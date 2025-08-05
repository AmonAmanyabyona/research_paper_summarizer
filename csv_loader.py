import pandas as pd

def load_csv(file_path):
    """Loads CSV file and returns list of row contents."""
    df = pd.read_csv(file_path)
    pages = [df.to_string(index=False)]  # Convert entire CSV into a text representation
    return pages

