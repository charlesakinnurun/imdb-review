"""
Configuration file for IMDB Sentiment Analysis project.
Contains all hyperparameters, paths, and settings.
"""

# File paths
DATA_PATH = "data/imdb.csv"
MODEL_SAVE_PATH = "models/"

# Data split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF Vectorizer parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.8

# Model parameters
MODELS_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'naive_bayes': {},
    'decision_tree': {
        'random_state': RANDOM_STATE
    }
}

# Hyperparameter tuning grid for Logistic Regression
LOGISTIC_REGRESSION_GRID = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'liblinear']
}

# Grid search parameters
GRID_SEARCH_CV = 3
GRID_SEARCH_N_JOBS = -1

# Visualization settings
FIGURE_SIZE = (10, 6)
HEATMAP_CMAP = "Greens"
