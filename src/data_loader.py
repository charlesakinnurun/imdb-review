"""
Data Loading Module
Handles loading the IMDB dataset and initial data exploration.
"""

import pandas as pd
from typing import Tuple


class DataLoader:
    """Class to handle data loading and initial exploration."""
    
    def __init__(self, file_path: str):
        """
        Initialize DataLoader with file path.
        
        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Dataset loaded successfully from {self.file_path}")
            print(f"  Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: '{self.file_path}' was not found. "
                "Please ensure the file is accessible."
            )
    
    def explore_data(self) -> None:
        """Print basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\n1. Dataset Info:")
        print("-" * 60)
        self.df.info()
        
        print("\n2. First few rows:")
        print("-" * 60)
        print(self.df.head())
        
        print("\n3. Dataset shape:")
        print("-" * 60)
        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")
        
        print("\n4. Missing values:")
        print("-" * 60)
        print(self.df.isnull().sum())
        
        print("\n5. Sentiment distribution:")
        print("-" * 60)
        print(self.df['sentiment'].value_counts())
        
        print("\n6. Basic statistics:")
        print("-" * 60)
        print(f"Total reviews: {len(self.df)}")
        print(f"Unique sentiments: {self.df['sentiment'].nunique()}")
        print(f"Average review length: {self.df['review'].str.len().mean():.2f} characters")
    
    def get_data(self) -> Tuple[pd.Series, pd.Series]:
        """
        Get features and target variables.
        
        Returns:
            Tuple of (reviews, sentiments)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df['review'], self.df['sentiment']


if __name__ == "__main__":
    # Test the module
    from config import DATA_PATH
    
    loader = DataLoader(DATA_PATH)
    loader.load_data()
    loader.explore_data()
    X, y = loader.get_data()
    print(f"\n✓ Extracted {len(X)} reviews and {len(y)} labels")
