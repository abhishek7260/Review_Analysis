import pandas as pd 
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

mlflow.set_tracking_uri("localhost:5000")
test_data=pd.read_csv("E:\review_classification\data\processed\processed_test.csv")
x_test=test_data.drop('Liked',axis=1)
y_test=test_data['Liked']
tfidf = TfidfVectorizer(max_features=5000)
x_test_tfidf = tfidf.fit_transform(x_test).toarray()

lr_model = pickle.load(open("results/lr_model.pkl", "rb"))
with mlflow.start_run(run_name="LogisticRegression_Classifier"):
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(x_train_tfidf, y_train)

    # Predictions and Metrics
    lr_predictions = lr_model.predict(x_test_tfidf)
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", lr_accuracy)
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

    print("\nLogistic Regression Classifier:")
    print(f"Accuracy: {lr_accuracy:.2f}")
    print(classification_report(y_test, lr_predictions))