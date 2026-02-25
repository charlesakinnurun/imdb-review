"""
Hyperparameter Tuning Module
Handles hyperparameter optimization using GridSearchCV.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple
import time


class HyperparameterTuner:
    """Class to handle hyperparameter tuning."""
    
    def __init__(
        self, 
        param_grid: Dict[str, list],
        cv: int = 3,
        n_jobs: int = -1,
        verbose: int = 2
    ):
        """
        Initialize HyperparameterTuner.
        
        Args:
            param_grid: Dictionary of hyperparameters to search
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all processors)
            verbose: Verbosity level
        """
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.grid_search = None
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def create_pipeline(
        self, 
        vectorizer: TfidfVectorizer,
        base_model: Any = None
    ) -> Pipeline:
        """
        Create a pipeline with vectorizer and classifier.
        
        Args:
            vectorizer: Fitted TfidfVectorizer
            base_model: Base model to use (default: LogisticRegression)
            
        Returns:
            sklearn Pipeline
        """
        if base_model is None:
            base_model = LogisticRegression(max_iter=1000)
        
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', base_model)
        ])
        
        return pipeline
    
    def tune_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        vectorizer: TfidfVectorizer
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features (text)
            y_train: Training labels
            vectorizer: Fitted TfidfVectorizer
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        print(f"Parameter grid:")
        for param, values in self.param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nCross-validation folds: {self.cv}")
        print(f"Total combinations to try: {np.prod([len(v) for v in self.param_grid.values()])}")
        
        # Create pipeline
        pipeline = self.create_pipeline(vectorizer)
        
        # Perform grid search
        print("\nStarting grid search...")
        start_time = time.time()
        
        self.grid_search = GridSearchCV(
            pipeline,
            self.param_grid,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring='accuracy'
        )
        
        self.grid_search.fit(X_train, y_train)
        
        tuning_time = time.time() - start_time
        
        # Get best results
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        
        print(f"\n{'='*60}")
        print("TUNING RESULTS")
        print("="*60)
        print(f"Best cross-validation score: {self.best_score:.4f}")
        print(f"\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nTuning time: {tuning_time:.2f} seconds")
        
        return self.best_model, self.best_params, self.best_score
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        Get detailed cross-validation results.
        
        Returns:
            Dictionary of CV results
        """
        if self.grid_search is None:
            raise ValueError("Grid search not performed. Call tune_model() first.")
        
        return self.grid_search.cv_results_
    
    def print_cv_results(self, top_n: int = 5) -> None:
        """
        Print top N parameter combinations from cross-validation.
        
        Args:
            top_n: Number of top results to print
        """
        if self.grid_search is None:
            raise ValueError("Grid search not performed. Call tune_model() first.")
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} PARAMETER COMBINATIONS")
        print("="*60)
        
        results = self.grid_search.cv_results_
        
        # Sort by mean test score
        indices = np.argsort(results['mean_test_score'])[::-1][:top_n]
        
        for rank, idx in enumerate(indices, 1):
            print(f"\nRank {rank}:")
            print(f"  Mean CV Score: {results['mean_test_score'][idx]:.4f} "
                  f"(+/- {results['std_test_score'][idx]:.4f})")
            print(f"  Parameters: {results['params'][idx]}")
    
    def get_best_model(self) -> Any:
        """
        Get the best model from tuning.
        
        Returns:
            Best fitted model
        """
        if self.best_model is None:
            raise ValueError("Tuning not performed. Call tune_model() first.")
        
        return self.best_model
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters from tuning.
        
        Returns:
            Dictionary of best parameters
        """
        if self.best_params is None:
            raise ValueError("Tuning not performed. Call tune_model() first.")
        
        return self.best_params
    
    def get_best_score(self) -> float:
        """
        Get the best cross-validation score.
        
        Returns:
            Best CV score
        """
        if self.best_score is None:
            raise ValueError("Tuning not performed. Call tune_model() first.")
        
        return self.best_score


if __name__ == "__main__":
    # Test the module
    from config import (
        DATA_PATH, TEST_SIZE, RANDOM_STATE, 
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF,
        LOGISTIC_REGRESSION_GRID, GRID_SEARCH_CV
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
    
    # Hyperparameter tuning
    tuner = HyperparameterTuner(
        LOGISTIC_REGRESSION_GRID,
        cv=GRID_SEARCH_CV
    )
    best_model, best_params, best_score = tuner.tune_model(
        X_train, 
        y_train,
        feature_engineer.get_vectorizer()
    )
    tuner.print_cv_results(top_n=3)
    
    print(f"\n✓ Hyperparameter tuning complete!")
