"""
Example Usage Scenarios
Demonstrates various ways to use the modular sentiment analysis system.
"""

from main import SentimentAnalysisPipeline
from config import DATA_PATH


def example_1_full_pipeline():
    """Example 1: Run the complete pipeline with all features."""
    print("="*60)
    print("EXAMPLE 1: Full Pipeline")
    print("="*60)
    
    pipeline = SentimentAnalysisPipeline(DATA_PATH)
    pipeline.run_full_pipeline(
        perform_tuning=True,
        create_visualizations=True,
        interactive_mode=False  # Set to True for interactive mode
    )


def example_2_quick_training():
    """Example 2: Quick training without tuning or visualizations."""
    print("="*60)
    print("EXAMPLE 2: Quick Training (No Tuning/Viz)")
    print("="*60)
    
    pipeline = SentimentAnalysisPipeline(DATA_PATH)
    pipeline.run_full_pipeline(
        perform_tuning=False,
        create_visualizations=False,
        interactive_mode=False
    )


def example_3_step_by_step():
    """Example 3: Run pipeline step-by-step with custom logic."""
    print("="*60)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("="*60)
    
    pipeline = SentimentAnalysisPipeline(DATA_PATH)
    
    # Step 1: Load and explore
    pipeline.load_data()
    
    # Step 2: Preprocess
    pipeline.preprocess_data()
    
    # Step 3: Feature engineering
    pipeline.engineer_features()
    
    # Step 4: Train models
    pipeline.train_models()
    
    # Step 5: Evaluate
    pipeline.evaluate_models()
    
    print("\n✓ Step-by-step execution complete!")


def example_4_custom_predictions():
    """Example 4: Train model and make custom predictions."""
    print("="*60)
    print("EXAMPLE 4: Custom Predictions")
    print("="*60)
    
    pipeline = SentimentAnalysisPipeline(DATA_PATH)
    
    # Run pipeline without interactive mode
    pipeline.run_full_pipeline(
        perform_tuning=False,
        create_visualizations=False,
        interactive_mode=False
    )
    
    # Make predictions on custom reviews
    sample_reviews = [
        "This movie was absolutely fantastic! Best film I've seen this year.",
        "Terrible waste of time. Boring plot and bad acting.",
        "It was okay, nothing special but watchable on a lazy Sunday.",
        "Mind-blowing cinematography and incredible performances!",
        "I fell asleep halfway through. Not recommended."
    ]
    
    print("\n" + "="*60)
    print("MAKING PREDICTIONS ON SAMPLE REVIEWS")
    print("="*60)
    
    pipeline.predict_new_reviews(sample_reviews)


def example_5_using_individual_modules():
    """Example 5: Use individual modules independently."""
    print("="*60)
    print("EXAMPLE 5: Using Individual Modules")
    print("="*60)
    
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from config import TEST_SIZE, RANDOM_STATE
    
    # Load data
    print("\n1. Loading data...")
    loader = DataLoader(DATA_PATH)
    loader.load_data()
    X, y = loader.get_data()
    print(f"✓ Loaded {len(X)} reviews")
    
    # Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor(TEST_SIZE, RANDOM_STATE)
    X_clean = preprocessor.clean_reviews(X)
    print(f"✓ Cleaned {len(X_clean)} reviews")
    
    # Show example
    print("\n3. Example transformation:")
    print(f"Original: {X.iloc[0][:100]}...")
    print(f"Cleaned:  {X_clean.iloc[0][:100]}...")


def example_6_visualization_only():
    """Example 6: Train models and create visualizations only."""
    print("="*60)
    print("EXAMPLE 6: Training + Visualization Focus")
    print("="*60)
    
    pipeline = SentimentAnalysisPipeline(DATA_PATH)
    
    # Run pipeline with visualization focus
    pipeline.run_full_pipeline(
        perform_tuning=False,  # Skip tuning for speed
        create_visualizations=True,  # Focus on visualizations
        interactive_mode=False
    )


def example_7_hyperparameter_tuning_focus():
    """Example 7: Focus on hyperparameter tuning."""
    print("="*60)
    print("EXAMPLE 7: Hyperparameter Tuning Focus")
    print("="*60)
    
    pipeline = SentimentAnalysisPipeline(DATA_PATH)
    
    # Load, preprocess, and engineer features
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.engineer_features()
    
    # Train initial models
    pipeline.train_models()
    pipeline.evaluate_models()
    
    # Focus on tuning
    print("\n" + "="*60)
    print("FOCUSING ON HYPERPARAMETER TUNING")
    print("="*60)
    
    pipeline.tune_best_model()
    pipeline.evaluate_tuned_model()


def main():
    """Main function to run examples."""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS - EXAMPLE USAGE SCENARIOS")
    print("="*60)
    
    examples = {
        '1': ('Full Pipeline', example_1_full_pipeline),
        '2': ('Quick Training', example_2_quick_training),
        '3': ('Step-by-Step', example_3_step_by_step),
        '4': ('Custom Predictions', example_4_custom_predictions),
        '5': ('Individual Modules', example_5_using_individual_modules),
        '6': ('Visualization Focus', example_6_visualization_only),
        '7': ('Tuning Focus', example_7_hyperparameter_tuning_focus),
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Run all examples")
    
    choice = input("\nSelect an example (0-7): ").strip()
    
    if choice == '0':
        for name, func in examples.values():
            print(f"\n\nRunning: {name}")
            print("="*60)
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    elif choice in examples:
        examples[choice][1]()
    else:
        print("Invalid choice. Running Example 1 (Full Pipeline)...")
        example_1_full_pipeline()


if __name__ == "__main__":
    # You can directly call any example function or run main() for interactive selection
    
    # Uncomment the example you want to run:
    # example_1_full_pipeline()
    # example_2_quick_training()
    # example_3_step_by_step()
    # example_4_custom_predictions()
    # example_5_using_individual_modules()
    # example_6_visualization_only()
    # example_7_hyperparameter_tuning_focus()
    
    # Or run interactive menu:
    main()
