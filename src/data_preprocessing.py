"""
Data Preprocessing Module
Handles text cleaning, label encoding, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import re


class DataPreprocessor:
    """Class to handle data preprocessing tasks."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataPreprocessor.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_reviews(self, reviews: pd.Series) -> pd.Series:
        """
        Clean all reviews in the series.
        
        Args:
            reviews: Series of review texts
            
        Returns:
            Series of cleaned reviews
        """
        print("\n" + "="*60)
        print("TEXT CLEANING")
        print("="*60)
        
        print("Cleaning reviews...")
        cleaned_reviews = reviews.apply(self.clean_text)
        
        print(f"✓ Cleaned {len(cleaned_reviews)} reviews")
        print(f"\nExample transformation:")
        print(f"Before: {reviews.iloc[0][:100]}...")
        print(f"After:  {cleaned_reviews.iloc[0][:100]}...")
        
        return cleaned_reviews
    
    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Encode sentiment labels to numerical values.
        
        Args:
            labels: Series of sentiment labels
            
        Returns:
            Numpy array of encoded labels
        """
        print("\n" + "="*60)
        print("LABEL ENCODING")
        print("="*60)
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print("Label mapping:")
        for idx, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {idx}")
        
        return encoded_labels
    
    def split_data(
        self, 
        X: pd.Series, 
        y: np.ndarray
    ) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature data (reviews)
            y: Target data (encoded labels)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training set size: {len(X_train)} ({(1-self.test_size)*100:.0f}%)")
        print(f"Testing set size:  {len(X_test)} ({self.test_size*100:.0f}%)")
        print(f"\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            original_label = self.label_encoder.inverse_transform([label])[0]
            print(f"  {original_label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def get_label_encoder(self) -> LabelEncoder:
        """
        Get the fitted label encoder.
        
        Returns:
            Fitted LabelEncoder object
        """
        return self.label_encoder


if __name__ == "__main__":
    # Test the module
    from config import TEST_SIZE, RANDOM_STATE, DATA_PATH
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(DATA_PATH)
    loader.load_data()
    X, y = loader.get_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(TEST_SIZE, RANDOM_STATE)
    X_clean = preprocessor.clean_reviews(X)
    y_encoded = preprocessor.encode_labels(y)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_clean, y_encoded)
    
    print(f"\n✓ Preprocessing complete!")
