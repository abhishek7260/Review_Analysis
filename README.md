# Restaurant Review Classification Project
## Overview

This project classifies restaurant reviews as either "Liked" or "Not Liked" using a machine learning pipeline. The pipeline includes data preprocessing, text vectorization using TF-IDF, training a Logistic Regression model, and tracking results with MLflow and DagsHub.

## Dependencies

Install the following Python libraries:

pip install pandas numpy scikit-learn nltk mlflow seaborn matplotlib dagshub

## Data Processing
  1. Data Splitting:
     - Raw data (Restaurant_Reviews.tsv) is split into training and testing datasets.
     - The training data is saved in data/raw/train.csv.
     - The training data is saved in data/raw/train.csv.
  2. Preprocessing Steps:
     - Remove special characters from reviews.
     - Eliminate stopwords using a custom stopwords list.
     - Apply stemming using the Porter Stemmer.
     - Save processed datasets in data/processed/.
## Model Training
 - Algorithm: Logistic Regression
 - Vectorization: TF-IDF (max features = 5000)
 - The model and vectorizer are saved as pickled files in the results/ directory
## Testing and Evaluation
 - Pre-trained model and vectorizer are loaded to evaluate the test dataset.
 - Metrics:
    - Accuracy
    - Precision
    - Recall
    - Confusion Matrix
## Experiment Tracking with MLflow
 - Experiment results are tracked using MLflow and DagsHub.
 - The experiment logs:
    - Model parameters
    - Metrics (accuracy, precision, recall)
    - Classification report
    - Confusion matrix
## Steps to Setup:
 1. Initialize DagsHub repository:

    **dagshub.init(repo_owner='abhishek7260', repo_name='Review_Analysis', mlflow=True)**
2. Set MLflow tracking URI:

    **mlflow.set_tracking_uri("https://dagshub.com/abhishek7260/Review_Analysis.mlflow")**

## License

This project is licensed under the MIT License. For details, see the LICENSE file.
