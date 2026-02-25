"""
Prediction Module
Handles making predictions on new review data.
"""

import numpy as np
from typing import Any, Tuple


class Predictor:
    """Class to handle predictions on new data."""
    
    def __init__(self, model: Any, label_encoder: Any):
        """
        Initialize Predictor.
        
        Args:
            model: Trained model (can be a pipeline)
            label_encoder: Fitted LabelEncoder
        """
        self.model = model
        self.label_encoder = label_encoder
    
    def predict_single(self, review: str) -> Tuple[str, float]:
        """
        Predict sentiment for a single review.
        
        Args:
            review: Text review string
            
        Returns:
            Tuple of (predicted_sentiment, confidence_score)
        """
        # Predict the encoded label
        prediction_encoded = self.model.predict([review])[0]
        
        # Convert back to original label
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence score
        prediction_proba = self.model.predict_proba([review])[0]
        confidence_score = prediction_proba[prediction_encoded]
        
        return prediction_label, confidence_score
    
    def predict_batch(self, reviews: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sentiments for multiple reviews.
        
        Args:
            reviews: List of review strings
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Predict encoded labels
        predictions_encoded = self.model.predict(reviews)
        
        # Convert to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get confidence scores
        predictions_proba = self.model.predict_proba(reviews)
        confidence_scores = np.max(predictions_proba, axis=1)
        
        return predictions, confidence_scores
    
    def interactive_prediction(self) -> None:
        """
        Interactive mode for predicting sentiments of user-input reviews.
        Continues until user types 'exit'.
        """
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION MODE")
        print("="*60)
        print("Enter movie reviews to predict their sentiment.")
        print("Type 'exit' to quit.\n")
        
        while True:
            # Get user input
            review = input("Your Review: ").strip()
            
            # Check for exit
            if review.lower() == 'exit':
                print("\nExiting prediction mode. Goodbye!")
                break
            
            # Check for empty input
            if not review:
                print("⚠ Please enter a non-empty review.\n")
                continue
            
            # Make prediction
            try:
                sentiment, confidence = self.predict_single(review)
                
                print(f"\n{'='*60}")
                print(f"Predicted Sentiment: {sentiment.upper()}")
                print(f"Confidence Score:    {confidence:.4f} ({confidence*100:.2f}%)")
                print("="*60 + "\n")
                
            except Exception as e:
                print(f"⚠ Error making prediction: {e}\n")
    
    def predict_and_display(self, reviews: list, show_review: bool = True) -> None:
        """
        Predict sentiments and display results in a formatted way.
        
        Args:
            reviews: List of review strings
            show_review: Whether to display the review text
        """
        predictions, confidences = self.predict_batch(reviews)
        
        print("\n" + "="*60)
        print("BATCH PREDICTIONS")
        print("="*60)
        
        for idx, (review, sentiment, confidence) in enumerate(zip(reviews, predictions, confidences), 1):
            print(f"\nReview {idx}:")
            if show_review:
                # Display truncated review
                max_length = 100
                display_review = review[:max_length] + "..." if len(review) > max_length else review
                print(f"  Text: {display_review}")
            print(f"  Sentiment: {sentiment.upper()}")
            print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print("\n" + "="*60)


class ModelPredictor:
    """
    Simplified predictor that handles text preprocessing internally
    when using a pipeline model.
    """
    
    def __init__(self, pipeline_model: Any, label_encoder: Any):
        """
        Initialize ModelPredictor with a pipeline model.
        
        Args:
            pipeline_model: Trained pipeline (includes vectorization)
            label_encoder: Fitted LabelEncoder
        """
        self.pipeline_model = pipeline_model
        self.label_encoder = label_encoder
    
    def predict(self, review: str) -> Tuple[str, float]:
        """
        Predict sentiment for a review (handles preprocessing).
        
        Args:
            review: Raw review text
            
        Returns:
            Tuple of (predicted_sentiment, confidence_score)
        """
        # The pipeline handles text preprocessing
        prediction_encoded = self.pipeline_model.predict([review])[0]
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        prediction_proba = self.pipeline_model.predict_proba([review])[0]
        confidence_score = prediction_proba[prediction_encoded]
        
        return prediction_label, confidence_score


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
    
    # Train a model
    trainer = ModelTrainer(MODELS_CONFIG)
    trained_models = trainer.train_all_models(X_train_tfidf, y_train)
    
    # Create predictor
    predictor = Predictor(
        trained_models['Logistic Regression'],
        preprocessor.get_label_encoder()
    )
    
    # Test with sample reviews
    sample_reviews = [
        "This movie was absolutely amazing! Best film I've seen this year.",
        "Terrible waste of time. Boring and predictable plot.",
        "It was okay, nothing special but watchable."
    ]
    
    predictor.predict_and_display(sample_reviews)
    
    print(f"\n✓ Prediction module test complete!")
