import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

# Load data
train_data = pd.read_csv("E:\\review_classification\\data\\processed\\processed_train.csv")

# Ensure x_train contains only text data
x_train = train_data['Review']  # Replace 'Review' with the actual column name for text
y_train = train_data['Liked']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)  
x_train_tfidf = tfidf.fit_transform(x_train).toarray()

# Train Logistic Regression model
svc_model = SVC()
svc_model.fit(x_train_tfidf, y_train)

pickle.dump(svc_model,open("results/svc_model.pkl","wb"))
pickle.dump(tfidf, open("results/tfidf_vectorizer.pkl", "wb"))