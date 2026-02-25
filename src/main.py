"""
Main Module
Orchestrates all components of the IMDB Sentiment Analysis pipeline.
"""

import warnings
import argparse
from typing import Optional

# Import all modules
from config import *
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from hyperparameter_tuning import HyperparameterTuner
from prediction import Predictor, ModelPredictor
from visualization import Visualizer

warnings.filterwarnings('ignore')


class SentimentAnalysisPipeline:
    """
    Complete pipeline for IMDB Sentiment Analysis.
    Orchestrates data loading, preprocessing, training, evaluation, and prediction.
    """
    
    def __init__(self, data_path: str = DATA_PATH):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to the dataset
        """
        self.data_path = data_path
        
        # Initialize components
        self.loader = None
        self.preprocessor = None
        self.feature_engineer = None
        self.trainer = None
        self.evaluator = None
        self.tuner = None
        self.visualizer = None
        self.predictor = None
        
        # Data storage
        self.X = None
        self.y = None
        self.X_clean = None
        self.y_encoded = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
        # Models storage
        self.trained_models = None
        self.best_model = None
        self.best_model_name = None
        self.tuned_model = None
    
    def run_full_pipeline(
        self, 
        perform_tuning: bool = True,
        create_visualizations: bool = True,
        interactive_mode: bool = False
    ) -> None:
        """
        Run the complete pipeline from start to finish.
        
        Args:
            perform_tuning: Whether to perform hyperparameter tuning
            create_visualizations: Whether to create visualizations
            interactive_mode: Whether to enter interactive prediction mode
        """
        print("\n" + "="*60)
        print("IMDB SENTIMENT ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Load Data
        self.load_data()
        
        # Step 2: Preprocess Data
        self.preprocess_data()
        
        # Step 3: Feature Engineering
        self.engineer_features()
        
        # Step 4: Train Models
        self.train_models()
        
        # Step 5: Evaluate Models
        self.evaluate_models()
        
        # Step 6: Hyperparameter Tuning (optional)
        if perform_tuning:
            self.tune_best_model()
            self.evaluate_tuned_model()
        
        # Step 7: Create Visualizations (optional)
        if create_visualizations:
            self.create_visualizations()
        
        # Step 8: Interactive Prediction (optional)
        if interactive_mode:
            self.run_interactive_prediction()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
    
    def load_data(self) -> None:
        """Step 1: Load the dataset."""
        print("\n" + "="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        self.loader = DataLoader(self.data_path)
        self.loader.load_data()
        self.loader.explore_data()
        self.X, self.y = self.loader.get_data()
    
    def preprocess_data(self) -> None:
        """Step 2: Preprocess the data."""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        self.preprocessor = DataPreprocessor(TEST_SIZE, RANDOM_STATE)
        self.X_clean = self.preprocessor.clean_reviews(self.X)
        self.y_encoded = self.preprocessor.encode_labels(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.preprocessor.split_data(self.X_clean, self.y_encoded)
    
    def engineer_features(self) -> None:
        """Step 3: Engineer features using TF-IDF."""
        print("\n" + "="*60)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*60)
        
        self.feature_engineer = FeatureEngineer(
            TFIDF_MAX_FEATURES,
            TFIDF_MIN_DF,
            TFIDF_MAX_DF
        )
        self.X_train_tfidf = self.feature_engineer.fit_transform(self.X_train)
        self.X_test_tfidf = self.feature_engineer.transform(self.X_test)
    
    def train_models(self) -> None:
        """Step 4: Train multiple models."""
        print("\n" + "="*60)
        print("STEP 4: MODEL TRAINING")
        print("="*60)
        
        self.trainer = ModelTrainer(MODELS_CONFIG)
        self.trained_models = self.trainer.train_all_models(
            self.X_train_tfidf,
            self.y_train
        )
    
    def evaluate_models(self) -> None:
        """Step 5: Evaluate all models."""
        print("\n" + "="*60)
        print("STEP 5: MODEL EVALUATION")
        print("="*60)
        
        self.evaluator = ModelEvaluator()
        self.evaluator.evaluate_all_models(
            self.trained_models,
            self.X_test_tfidf,
            self.y_test,
            self.preprocessor.get_label_encoder()
        )
        self.evaluator.print_detailed_results(
            self.preprocessor.get_label_encoder()
        )
        self.evaluator.compare_models()
        
        # Get best model
        self.best_model_name, self.best_model, _ = \
            self.evaluator.get_best_model(self.trained_models)
    
    def tune_best_model(self) -> None:
        """Step 6: Tune the best model using GridSearchCV."""
        print("\n" + "="*60)
        print("STEP 6: HYPERPARAMETER TUNING")
        print("="*60)
        
        self.tuner = HyperparameterTuner(
            LOGISTIC_REGRESSION_GRID,
            cv=GRID_SEARCH_CV,
            n_jobs=GRID_SEARCH_N_JOBS
        )
        
        self.tuned_model, _, _ = self.tuner.tune_model(
            self.X_train,
            self.y_train,
            self.feature_engineer.get_vectorizer()
        )
        
        self.tuner.print_cv_results(top_n=3)
    
    def evaluate_tuned_model(self) -> None:
        """Evaluate the tuned model."""
        print("\n" + "="*60)
        print("TUNED MODEL EVALUATION")
        print("="*60)
        
        # Create a temporary evaluator for the tuned model
        temp_evaluator = ModelEvaluator()
        temp_evaluator.evaluate_model(
            "Tuned Logistic Regression",
            self.tuned_model,
            self.X_test,
            self.y_test,
            self.preprocessor.get_label_encoder()
        )
        
        # Print detailed results
        print("\n" + "="*60)
        print("TUNED MODEL RESULTS")
        print("="*60)
        result = temp_evaluator.get_results()["Tuned Logistic Regression"]
        print(f"Accuracy: {result['accuracy']:.4f}")
        
        # Compare with original best model
        original_accuracy = self.evaluator.get_results()[self.best_model_name]['accuracy']
        print(f"\nComparison:")
        print(f"  Original {self.best_model_name}: {original_accuracy:.4f}")
        print(f"  Tuned Model: {result['accuracy']:.4f}")
        print(f"  Improvement: {(result['accuracy'] - original_accuracy):.4f}")
    
    def create_visualizations(self) -> None:
        """Step 7: Create visualizations."""
        print("\n" + "="*60)
        print("STEP 7: CREATING VISUALIZATIONS")
        print("="*60)
        
        self.visualizer = Visualizer(FIGURE_SIZE)
        
        # Sentiment distribution
        print("\n1. Plotting sentiment distribution...")
        self.visualizer.plot_sentiment_distribution(self.y)
        
        # Model comparison
        print("2. Plotting model comparison...")
        comparison_df = self.evaluator.compare_models()
        self.visualizer.plot_model_comparison(comparison_df)
        
        # Confusion matrices
        print("3. Plotting confusion matrices...")
        confusion_matrices = {
            name: self.evaluator.get_confusion_matrix(name)
            for name in self.trained_models.keys()
        }
        self.visualizer.plot_multiple_confusion_matrices(
            confusion_matrices,
            self.preprocessor.get_label_encoder().classes_
        )
        
        # Training times
        print("4. Plotting training times...")
        self.visualizer.plot_training_times(
            self.trainer.get_training_times()
        )
        
        # Classification metrics
        print("5. Plotting classification metrics...")
        self.visualizer.plot_classification_metrics(
            self.evaluator.get_results(),
            self.preprocessor.get_label_encoder()
        )
    
    def run_interactive_prediction(self) -> None:
        """Step 8: Run interactive prediction mode."""
        print("\n" + "="*60)
        print("STEP 8: INTERACTIVE PREDICTION")
        print("="*60)
        
        # Use tuned model if available, otherwise use best model
        model_to_use = self.tuned_model if self.tuned_model else self.best_model
        
        if self.tuned_model:
            # Tuned model is a pipeline
            self.predictor = ModelPredictor(
                model_to_use,
                self.preprocessor.get_label_encoder()
            )
        else:
            # Regular model needs manual preprocessing
            self.predictor = Predictor(
                model_to_use,
                self.preprocessor.get_label_encoder()
            )
        
        self.predictor.interactive_prediction()
    
    def predict_new_reviews(self, reviews: list) -> None:
        """
        Predict sentiments for a list of new reviews.
        
        Args:
            reviews: List of review strings
        """
        if self.predictor is None:
            # Create predictor with best available model
            model_to_use = self.tuned_model if self.tuned_model else self.best_model
            
            if self.tuned_model:
                self.predictor = ModelPredictor(
                    model_to_use,
                    self.preprocessor.get_label_encoder()
                )
            else:
                self.predictor = Predictor(
                    model_to_use,
                    self.preprocessor.get_label_encoder()
                )
        
        self.predictor.predict_and_display(reviews)


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description='IMDB Sentiment Analysis Pipeline'
    )
    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Skip hyperparameter tuning'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualizations'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enter interactive prediction mode after training'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=DATA_PATH,
        help='Path to the dataset'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SentimentAnalysisPipeline(args.data_path)
    pipeline.run_full_pipeline(
        perform_tuning=not args.no_tuning,
        create_visualizations=not args.no_viz,
        interactive_mode=args.interactive
    )


if __name__ == "__main__":
    main()
