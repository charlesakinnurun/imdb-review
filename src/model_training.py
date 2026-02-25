"""
Model Training Module
Handles training of multiple classification models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Any
import time


class ModelTrainer:
    """Class to handle training of multiple models."""
    
    def __init__(self, models_config: Dict[str, Dict[str, Any]]):
        """
        Initialize ModelTrainer with model configurations.
        
        Args:
            models_config: Dictionary of model configurations
        """
        self.models_config = models_config
        self.models = {}
        self.training_times = {}
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create model instances based on configuration.
        
        Returns:
            Dictionary of model name to model instance
        """
        print("\n" + "="*60)
        print("MODEL CREATION")
        print("="*60)
        
        models = {}
        
        if 'logistic_regression' in self.models_config:
            config = self.models_config['logistic_regression']
            models['Logistic Regression'] = LogisticRegression(**config)
            print("✓ Created Logistic Regression model")
        
        if 'naive_bayes' in self.models_config:
            config = self.models_config['naive_bayes']
            models['Naive Bayes'] = MultinomialNB(**config)
            print("✓ Created Naive Bayes model")
        
        if 'decision_tree' in self.models_config:
            config = self.models_config['decision_tree']
            models['Decision Tree'] = DecisionTreeClassifier(**config)
            print("✓ Created Decision Tree model")
        
        self.models = models
        return models
    
    def train_model(
        self, 
        name: str, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Any:
        """
        Train a single model.
        
        Args:
            name: Model name
            model: Model instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times[name] = training_time
        
        print(f"✓ {name} trained in {training_time:.2f} seconds")
        
        return model
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        if not self.models:
            self.create_models()
        
        trained_models = {}
        
        for name, model in self.models.items():
            trained_model = self.train_model(name, model, X_train, y_train)
            trained_models[name] = trained_model
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print("="*60)
        for name, train_time in self.training_times.items():
            print(f"{name:25s}: {train_time:6.2f} seconds")
        
        return trained_models
    
    def get_trained_models(self) -> Dict[str, Any]:
        """
        Get the dictionary of trained models.
        
        Returns:
            Dictionary of trained models
        """
        return self.models
    
    def get_training_times(self) -> Dict[str, float]:
        """
        Get the training times for all models.
        
        Returns:
            Dictionary of model names to training times
        """
        return self.training_times


if __name__ == "__main__":
    # Test the module
    from config import (
        DATA_PATH, TEST_SIZE, RANDOM_STATE, 
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF,
        MODELS_CONFIG
    )
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
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
    
    # Train models
    trainer = ModelTrainer(MODELS_CONFIG)
    trained_models = trainer.train_all_models(X_train_tfidf, y_train)
    
    print(f"\n✓ All models trained successfully!")
