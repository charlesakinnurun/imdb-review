"""
Feature Engineering Module
Handles text vectorization using TF-IDF.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from scipy.sparse import csr_matrix


class FeatureEngineer:
    """Class to handle feature engineering for text data."""
    
    def __init__(
        self, 
        max_features: int = 5000, 
        min_df: int = 5, 
        max_df: float = 0.8
    ):
        """
        Initialize FeatureEngineer with TF-IDF parameters.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
    
    def fit_transform(self, texts: pd.Series) -> csr_matrix:
        """
        Fit the TF-IDF vectorizer and transform the texts.
        
        Args:
            texts: Series of text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING (TF-IDF)")
        print("="*60)
        
        print(f"Creating TF-IDF vectorizer with:")
        print(f"  Max features: {self.max_features}")
        print(f"  Min document frequency: {self.min_df}")
        print(f"  Max document frequency: {self.max_df}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )
        
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        print(f"\n✓ TF-IDF transformation complete")
        print(f"  Shape: {X_tfidf.shape}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")
        
        return X_tfidf
    
    def transform(self, texts: pd.Series) -> csr_matrix:
        """
        Transform texts using the fitted vectorizer.
        
        Args:
            texts: Series of text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        
        X_tfidf = self.vectorizer.transform(texts)
        
        print(f"\n✓ Transformed {X_tfidf.shape[0]} documents")
        print(f"  Feature shape: {X_tfidf.shape}")
        
        return X_tfidf
    
    def get_feature_names(self) -> list:
        """
        Get the feature names from the vectorizer.
        
        Returns:
            List of feature names
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n: int = 20) -> list:
        """
        Get the top N features with highest average TF-IDF scores.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of tuples (feature_name, score)
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        
        feature_names = self.get_feature_names()
        return feature_names[:n]
    
    def get_vectorizer(self) -> TfidfVectorizer:
        """
        Get the fitted vectorizer.
        
        Returns:
            Fitted TfidfVectorizer object
        """
        return self.vectorizer


if __name__ == "__main__":
    # Test the module
    from config import (
        DATA_PATH, TEST_SIZE, RANDOM_STATE, 
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF
    )
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader(DATA_PATH)
    loader.load_data()
    X, y = loader.get_data()
    
    preprocessor = DataPreprocessor(TEST_SIZE, RANDOM_STATE)
    X_clean = preprocessor.clean_reviews(X)
    y_encoded = preprocessor.encode_labels(y)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_clean, y_encoded)
    
    # Feature engineering
    feature_engineer = FeatureEngineer(
        TFIDF_MAX_FEATURES, 
        TFIDF_MIN_DF, 
        TFIDF_MAX_DF
    )
    X_train_tfidf = feature_engineer.fit_transform(X_train)
    X_test_tfidf = feature_engineer.transform(X_test)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"\nTop 10 features:")
    for feature in feature_engineer.get_top_features(10):
        print(f"  - {feature}")
