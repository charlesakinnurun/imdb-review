# %% [markdown]
# # IMBD Review Sentinment Analysis

# %% [markdown]
# ### Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
from config import DATA_PATH

# %%
# Supress all future warnings to keep the output clean
warnings.filterwarnings("ignore")

# %% [markdown]
# ### Load and Initial Data Exploration

# %%
# Load the dataset (using the file provided in the environment)
# The dataset cotains movie reviews and their corresponding sentiment (positive/negative)
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' was not found. Please ensure the file is accessible")
    exit()

# %%
# Display the Data Information
print(df.info())

# %%
df

# %%
print("===== Target Distribution (Sentiment) Before Training =====")

sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Sentiment Distribution (Before Training) - Pie Chart")
plt.axis("equal")  # Ensures the pie is circular
plt.show()

# %%
# Check the distribution of the target variable (sentiment)
# This is cruical to identify if the dataset is balanced or imbalanced
print("===== Target Distribution (Sentiment) Before Training =====")
print(df["sentiment"].value_counts())
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment",data=df)
plt.title("Distribution of Sentinments (Before Training)")
plt.ylim(10000,25000) # Set y-axis limits for better visualization
plt.show() # Visualization: Before Training -  Target Distribution

# %% [markdown]
# ### Data Preprocessing

# %%
# Number of elemnents in the data
print("===== Number of Elements =====")
len(df)

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("===== Missing Values =====")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("===== Duplicated Rows =====")
print(df_duplicated)

# %%
# Drop duplicated rows
df.drop_duplicates(inplace=True)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("===== Duplicated Rows =====")
print(df_duplicated)

# %% [markdown]
# ### Pre-Training Visualization

# %%
# Check the distribution of the target variable (sentiment)
# This is cruical to identify if the dataset is balanced or imbalanced
print("===== Target Distribution (Sentiment) Before Training =====")
print(df["sentiment"].value_counts())
plt.Figure(figsize=(6,4))
sns.countplot(x="sentiment",data=df)
plt.title("Distribution of Sentinments (Before Training)")
plt.ylim(24000,25000) # Set y-axis limits for better visualization
plt.show() # Visualization: Before Training -  Target Distribution

# %% [markdown]
# ### Data Encoding

# %%
# Encode the categorical target variable ("sentinment") into numerical format (0 and 1)
# "positive" -> 1, "Negative" -> 0
le = LabelEncoder()
df["sentiment_encoded"] = le.fit_transform(df["sentiment"])
# The classes are encoded are mapped: 0 is "negative", 1 is "positive"
print(f"Encoded Classes: {dict(zip(le.classes_,le.transform(le.classes_)))}")

# %% [markdown]
# ### Feature Engineering

# %%
#  Define features (X -the "review" text) and target (y - the "sentinment_encoded")
X = df["review"]
y = df["sentiment_encoded"]

# %% [markdown]
# ### Data Splitting

# %%
# Split the data into training and testing sets
# The test size is 20% (random_state=42 ensures reproducibily)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# ### Model Pipeline Setup and Comparison

# %%
# We will compare three common classification algorithms:
# 1. Logistic Regression (Simple, highly interpretable linear model)
# 2. Decision Tree (Non-linear,captures complex rules, but prone to overfitting)
# 3. Multinominal Naive Bayes (Probabilistic, excellent for text classification due to TF-IDF features)

# A Pipeline is used to chain the preprocessing step (TF-IDF Vectorization) and the classifier
# TF-IDF (Term Frequency-Inverse Document Frequency) converts text into a numerical feature matrix
# It assigns a weight to each word based on its frequency in a document relative to its frequency across all documents

pipelines = {
    "LogisticRegression":Pipeline([
        ("tfidf",TfidfVectorizer(stop_words="english")),
        ("clf",LogisticRegression(random_state=42))
    ]),
    "DecisionTree":Pipeline([
        ("tfidf",TfidfVectorizer(stop_words="english")),
        ("clf",DecisionTreeClassifier(random_state=42))
    ]),
    "MultinomialNB":Pipeline([
        ("tfidf",TfidfVectorizer(stop_words="english")),
        ("clf",MultinomialNB())
    ])
}

# %%
# Dictionary to store performance metris for comparison
results = {}

# %%
# Iterate through each pipeline,train the model and evaluate its performance
print("----- Initial Model Training and Evaluation (Before Tuning) -----")
for name,pipeline in pipelines.items():
    print(f"Training {name}......")

    # Train the pipeline: Vectorization and classification are done in one step
    pipeline.fit(X_train,y_train)

    # Predict the target on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate and store the accuracy score
    accuracy = accuracy_score(y_test,y_pred)
    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")
    # Show the full classification report (precison,recall,f1-score)
    print(classification_report(y_test,y_pred,target_names=["negative","positive"]))

    # Visualization: Confusion Matrix for initial models
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["negative","positive"],
                yticklabels=["negative","positive"])
    plt.title(f"Confusion Matrix for {name} (Initial)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show() # Visualization: Initial Confusion Matrix

# %% [markdown]
# ### Hyperparameter Tuning for Logistic Regression

# %%
# Defin the parameter grid for Logistic Regression
# "tfidf__ngram_range": The range of word combinations to use (e.g, single words, or pairs of words)
# "clf__C": Inverse of regularization strength (smaller values specify stronger regularization)
param_grid_lr = {
    "tfidf__ngram_range":[(1,1),(1,2)],  # Unigrams (1,1) or Unigrams and Biagrams (1,2)
    "clf__C":[0.1,1,10],  # Regularization strength C
    "clf__solver":["liblinear"]
}

# %%
#  Create a new pipeline for tuning
lr_pipeline = Pipeline([
    ("tfidf",TfidfVectorizer(stop_words="english")),
    ("clf",LogisticRegression(random_state=42,max_iter=1000)) # Increased max_iter for convergence
])

# %%
# Setup GridSearchCV with 3-fold cross-valiadtion
grid_search_lr = GridSearchCV(lr_pipeline,param_grid_lr,cv=3,verbose=3,n_jobs=-1,scoring="accuracy")

# %%
# Execute the grid search on the training data
grid_search_lr.fit(X_train,y_train)

# %%
# Store the best model and its score
best_lr_model = grid_search_lr.best_estimator_
best_lr_score = grid_search_lr.best_score_

# %%
print(f"Best Cross-validation Accuracy (Logistic Regression): {best_lr_score:.4f}")
print(f"Best Parameters Found: {grid_search_lr.best_params_}")

# %% [markdown]
# ### Final Evaluation of Tuned Model

# %%
# Evaluate the best model on the unseen test data
y_pred_tuned = best_lr_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test,y_pred_tuned)

print("----- Final Evauation of Hyperparameter-Tuned Logistic Regression -----")
print(f"Tuned Logistic Regression Test Accuracy: {tuned_accuracy:.4f}")
print(classification_report(y_test,y_pred_tuned,target_names=["negative","positive"]))

# %%
# Visualization: Final Comparison of Model Accuracies
results['Tuned Logistic Regression'] = tuned_accuracy

plt.figure(figsize=(10,6))
model_names = list(results.keys())
accuracies = list(results.values())
sns.barplot(x=model_names,y=accuracies,palette="viridis")
plt.ylim(0.8,1.0) # Set reasonable limits for classification accuracy
plt.title("Model Comparison: Accuracy Before and After Tuning")
plt.ylabel("Accuracy")
plt.xticks(rotation=45,ha="right")
plt.grid(axis="y",linestyle="--")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show() # Visualization After Training - Model Comparison

# %%
# Visualization: Confusion Matrix for the final best model
cm_tuned = confusion_matrix(y_test,y_pred_tuned)
plt.Figure(figsize=(5,4))
sns.heatmap(cm_tuned,annot=True,fmt="d",cmap="Greens",
            xticklabels=["negative","positive"],
            yticklabels=["negative","positive"])
plt.title("Confusion Matrix for Tuned Logiistic Regression (Final)")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show() # Visualization: Final Tuned Confusion Matrix

# %% [markdown]
# ### Function for New Prediction

# %%
# Define a function to take a new review and predict its sentiment using the best model
def predict_new_review(model):
    print("----- New Prediction Input -----")
    print("Enter a movie review to predict its sentiment (type 'exit' to quit):")

    while True:
        # Get user input for a new review
        new_review = input("Your Review:")
        if new_review.lower() == "exit":
            break
        if not new_review.strip():
            print("Please enter a non-empty review.")
            continue

        # Predict the sentinment. The model handles the TF-IDF vectorization automatically.
        prediction_encoded = model.predict([new_review])[0]

        # Convert the numerical prediction back to the original label (positive or negative)
        prediction_label = le.inverse_transform([prediction_encoded])[0]

        # Get the probability (confidence score) of the prediction
        # The probabiliity of the predicted class is [:,predicted_encoded]
        # [0] is used because we only pass one review
        prediction_proba = model.predict_proba([new_review])[0][prediction_encoded]

        print(f"Predicted Sentinment: {prediction_label.upper()}")
        print(f"Confidence Score: {prediction_proba:.4f}")


# Run the prediction function with best-tuned model
predict_new_review(best_lr_model)


