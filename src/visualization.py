"""
Visualization Module
Handles all plotting and visualization tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class Visualizer:
    """Class to handle all visualizations."""
    
    def __init__(self, figure_size: tuple = (10, 6)):
        """
        Initialize Visualizer.
        
        Args:
            figure_size: Default figure size for plots
        """
        self.figure_size = figure_size
        sns.set_style("whitegrid")
    
    def plot_sentiment_distribution(
        self, 
        sentiments: pd.Series,
        title: str = "Sentiment Distribution"
    ) -> None:
        """
        Plot the distribution of sentiments.
        
        Args:
            sentiments: Series of sentiment labels
            title: Plot title
        """
        plt.figure(figsize=self.figure_size)
        
        # Count sentiments
        sentiment_counts = sentiments.value_counts()
        
        # Create bar plot
        ax = sentiment_counts.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts):
            ax.text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(
        self, 
        comparison_df: pd.DataFrame,
        metric: str = 'Accuracy'
    ) -> None:
        """
        Plot comparison of models based on a metric.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot (default: Accuracy)
        """
        plt.figure(figsize=self.figure_size)
        
        # Create bar plot
        ax = plt.bar(
            comparison_df['Model'], 
            comparison_df[metric],
            color=['#3498db', '#e74c3c', '#2ecc71']
        )
        
        plt.title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim([comparison_df[metric].min() - 0.05, 1.0])
        
        # Add value labels on bars
        for i, (model, value) in enumerate(zip(comparison_df['Model'], comparison_df[metric])):
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray,
        labels: list,
        title: str = "Confusion Matrix",
        cmap: str = "Blues"
    ) -> None:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            labels: List of class labels
            title: Plot title
            cmap: Color map for heatmap
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_training_times(self, training_times: Dict[str, float]) -> None:
        """
        Plot training times for different models.
        
        Args:
            training_times: Dictionary of model names to training times
        """
        plt.figure(figsize=self.figure_size)
        
        models = list(training_times.keys())
        times = list(training_times.values())
        
        # Create bar plot
        ax = plt.bar(models, times, color=['#9b59b6', '#f39c12', '#1abc9c'])
        
        plt.title('Model Training Times', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (model, time) in enumerate(zip(models, times)):
            plt.text(i, time + 0.1, f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_confusion_matrices(
        self, 
        confusion_matrices: Dict[str, np.ndarray],
        labels: list,
        cmap: str = "Blues"
    ) -> None:
        """
        Plot multiple confusion matrices in a grid.
        
        Args:
            confusion_matrices: Dictionary of model names to confusion matrices
            labels: List of class labels
            cmap: Color map for heatmap
        """
        n_models = len(confusion_matrices)
        
        # Determine grid layout
        if n_models <= 3:
            rows, cols = 1, n_models
            figsize = (6 * n_models, 5)
        else:
            rows = (n_models + 1) // 2
            cols = 2
            figsize = (12, 5 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap=cmap,
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[idx],
                cbar=True
            )
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_classification_metrics(
        self, 
        results: Dict[str, Dict[str, Any]],
        label_encoder: Any
    ) -> None:
        """
        Plot precision, recall, and F1-score for all models.
        
        Args:
            results: Dictionary of evaluation results
            label_encoder: LabelEncoder for class labels
        """
        # Prepare data
        metrics_data = []
        
        for model_name, result in results.items():
            report = result['classification_report']
            for class_label in label_encoder.classes_:
                if class_label in report:
                    metrics_data.append({
                        'Model': model_name,
                        'Class': class_label,
                        'Precision': report[class_label]['precision'],
                        'Recall': report[class_label]['recall'],
                        'F1-Score': report[class_label]['f1-score']
                    })
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#e74c3c']
        
        for idx, metric in enumerate(metrics):
            for class_idx, class_label in enumerate(label_encoder.classes_):
                class_data = df[df['Class'] == class_label]
                axes[idx].bar(
                    [i + class_idx * 0.35 for i in range(len(class_data))],
                    class_data[metric],
                    width=0.35,
                    label=class_label,
                    color=colors[class_idx]
                )
            
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Model', fontsize=10)
            axes[idx].set_ylabel(metric, fontsize=10)
            axes[idx].set_xticks([i + 0.175 for i in range(len(results))])
            axes[idx].set_xticklabels(list(results.keys()), rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_review_length_distribution(
        self, 
        reviews: pd.Series,
        title: str = "Review Length Distribution"
    ) -> None:
        """
        Plot the distribution of review lengths.
        
        Args:
            reviews: Series of review texts
            title: Plot title
        """
        plt.figure(figsize=self.figure_size)
        
        # Calculate review lengths
        review_lengths = reviews.str.len()
        
        # Create histogram
        plt.hist(review_lengths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        plt.axvline(review_lengths.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {review_lengths.mean():.0f}')
        plt.axvline(review_lengths.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {review_lengths.median():.0f}')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Review Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()


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
    from model_evaluation import ModelEvaluator
    
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
    
    # Train and evaluate models
    trainer = ModelTrainer(MODELS_CONFIG)
    trained_models = trainer.train_all_models(X_train_tfidf, y_train)
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(
        trained_models, 
        X_test_tfidf, 
        y_test,
        preprocessor.get_label_encoder()
    )
    
    # Create visualizations
    visualizer = Visualizer()
    
    print("\n✓ Creating visualizations...")
    
    # Plot sentiment distribution
    visualizer.plot_sentiment_distribution(y)
    
    # Plot model comparison
    comparison_df = evaluator.compare_models()
    visualizer.plot_model_comparison(comparison_df)
    
    # Plot confusion matrices
    confusion_matrices = {
        name: evaluator.get_confusion_matrix(name)
        for name in trained_models.keys()
    }
    visualizer.plot_multiple_confusion_matrices(
        confusion_matrices,
        preprocessor.get_label_encoder().classes_
    )
    
    print(f"\n✓ Visualization module test complete!")
