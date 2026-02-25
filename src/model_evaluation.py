"""
Model Evaluation Module
Handles model evaluation and comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from typing import Dict, Any, Tuple


class ModelEvaluator:
    """Class to handle model evaluation and comparison."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.results = {}
        self.predictions = {}
        self.confusion_matrices = {}
    
    def evaluate_model(
        self, 
        name: str, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        label_encoder: Any = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            name: Model name
            model: Trained model
            X_test: Test features
            y_test: Test labels
            label_encoder: LabelEncoder for converting labels back
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        if label_encoder:
            target_names = label_encoder.classes_
        else:
            target_names = None
        
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=target_names,
            output_dict=True
        )
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        self.predictions[name] = y_pred
        self.confusion_matrices[name] = cm
        
        print(f"✓ {name} - Accuracy: {accuracy:.4f}")
        
        return self.results[name]
    
    def evaluate_all_models(
        self, 
        models: Dict[str, Any], 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        label_encoder: Any = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            label_encoder: LabelEncoder for converting labels back
            
        Returns:
            Dictionary of evaluation results for all models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        for name, model in models.items():
            self.evaluate_model(name, model, X_test, y_test, label_encoder)
        
        return self.results
    
    def print_detailed_results(self, label_encoder: Any = None) -> None:
        """
        Print detailed evaluation results for all models.
        
        Args:
            label_encoder: LabelEncoder for converting labels back
        """
        print("\n" + "="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        
        for name, result in self.results.items():
            print(f"\n{name}")
            print("-" * 60)
            print(f"Accuracy: {result['accuracy']:.4f}")
            
            print("\nClassification Report:")
            report = result['classification_report']
            
            # Print per-class metrics
            if label_encoder:
                classes = label_encoder.classes_
            else:
                classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            for cls in classes:
                if cls in report:
                    metrics = report[cls]
                    print(f"{cls:<15} {metrics['precision']:<12.4f} "
                          f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
                          f"{int(metrics['support']):<10}")
            
            # Print averages
            print(f"\n{'Macro Avg':<15} {report['macro avg']['precision']:<12.4f} "
                  f"{report['macro avg']['recall']:<12.4f} "
                  f"{report['macro avg']['f1-score']:<12.4f}")
            print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<12.4f} "
                  f"{report['weighted avg']['recall']:<12.4f} "
                  f"{report['weighted avg']['f1-score']:<12.4f}")
            
            print("\nConfusion Matrix:")
            print(result['confusion_matrix'])
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models based on accuracy and create a summary.
        
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        for name, result in self.results.items():
            report = result['classification_report']
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision (macro)': report['macro avg']['precision'],
                'Recall (macro)': report['macro avg']['recall'],
                'F1-Score (macro)': report['macro avg']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_best_model(self, models: Dict[str, Any]) -> Tuple[str, Any, float]:
        """
        Get the best performing model based on accuracy.
        
        Args:
            models: Dictionary of trained models
            
        Returns:
            Tuple of (model_name, model_instance, accuracy)
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all_models() first.")
        
        best_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_name]['accuracy']
        best_model = models[best_name]
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print("="*60)
        
        return best_name, best_model, best_accuracy
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all evaluation results.
        
        Returns:
            Dictionary of evaluation results
        """
        return self.results
    
    def get_predictions(self, model_name: str) -> np.ndarray:
        """
        Get predictions for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Array of predictions
        """
        return self.predictions.get(model_name)
    
    def get_confusion_matrix(self, model_name: str) -> np.ndarray:
        """
        Get confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Confusion matrix
        """
        return self.confusion_matrices.get(model_name)


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
    from model_training import ModelTrainer
    
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
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(
        trained_models, 
        X_test_tfidf, 
        y_test,
        preprocessor.get_label_encoder()
    )
    evaluator.print_detailed_results(preprocessor.get_label_encoder())
    evaluator.compare_models()
    best_name, best_model, best_accuracy = evaluator.get_best_model(trained_models)
    
    print(f"\n✓ Evaluation complete!")
