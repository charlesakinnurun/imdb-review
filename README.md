# Imdb Movie Review
![Movie Review](/image.jpg)
The project is a sentiment-analysis implementation applied to movie review text, likely using the Internet Movie Database (IMDb) review dataset.
It contains data of labelled reviews (positive/negative), a Jupyter notebook for exploration and model development, and a Python script to run or deploy the trained model.
The workflow includes text-preprocessing (cleaning, tokenising), feature engineering (vectorising text), training a classification model and evaluating its performance.
It serves as a practical example of applying machine learning to natural language data, especially text classification tasks.
While useful as a learning resource, the documentation is minimal and the codebase appears to be designed at an introductory or educational level.

## Procedures
- Data Loading
- Data Exploration
- Data Preprocessing
- Pre-Training Visualization

![pre-training-visualization](/output1.png)
- Data Encoding
- Feature Engineering
- Data Splitting
- Model Pipeline 
- Model Comparison
- Hyperparameter Tuning
- Model Evaluation of Tuned Model

![confusion-matrix-logistic-regression](/output2.png)
![confusion-matrix-decison-tree-regression](/output3.png)
![confusion-matrix-multinomial-naive-bayes](/output4.png)
- Post-Training Visualization

![model-comparison-accuracy-before-and-after](/output5.png)
![confusion-matrix-tuned-logistic-regression](/output6.png)
- Function for New Prediction


## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/imdb-movie-review.git
cd imdb-movie-review
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
customer-personality/
│
├── model.ipynb  
|── model.py    
|── imdb.csv  
├── requirements.txt 
├── .gitignore
├── LICENSE
├── image.jpg       
├── output1.png     
├── output2.png     
├── output3.png     
├── output4.png     
├── output5.png     
├── output6.png     
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
└── README.md 
```

