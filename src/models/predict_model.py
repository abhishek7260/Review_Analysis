import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import dagshub

dagshub.init(repo_owner='abhishek7260', repo_name='Review_Analysis', mlflow=True)

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("https://dagshub.com/abhishek7260/Review_Analysis.mlflow")
mlflow.set_experiment("Support Vector Classifier review result")

# Load the test data
test_data = pd.read_csv("E:\\review_classification\\data\\processed\\processed_test.csv")

# Ensure x_test only contains the review text column (assuming 'Review' is the column name for text)
x_test = test_data['Review']
y_test = test_data['Liked']

# Ensure y_test is 1D
y_test = y_test.values.flatten()

# Load the pre-trained TfidfVectorizer and model
tfidf = pickle.load(open("results/tfidf_vectorizer.pkl", "rb"))
x_test_tfidf = tfidf.transform(x_test).toarray()
svc_model = pickle.load(open("results/svc_model.pkl", "rb"))

# Start an MLflow run
with mlflow.start_run(run_name="SupportVector_Classifier"):
    # Predictions
    svc_predictions = svc_model.predict(x_test_tfidf)

    # Metrics
    svc_accuracy = accuracy_score(y_test, svc_predictions)
    svc_precision = precision_score(y_test, svc_predictions)
    svc_recall = recall_score(y_test, svc_predictions)
    svc_conf_matrix = confusion_matrix(y_test, svc_predictions)

    # Log parameters, metrics, and artifacts
    mlflow.log_param("model_type", "SupportVector Classifier")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", svc_accuracy)
    mlflow.log_metric("precision", svc_precision)
    mlflow.log_metric("recall", svc_recall)

    # Log the classification report
    report = classification_report(y_test, svc_predictions, output_dict=True)
    mlflow.log_dict(report, "classification_report.json")

    # Log confusion matrix as an artifact
    plt.figure(figsize=(6, 6))
    sns.heatmap(svc_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Liked", "Liked"], yticklabels=["Not Liked", "Liked"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the dataset
    test_data.to_csv("processed_test_logged.csv", index=False)
    mlflow.log_artifact("processed_test_logged.csv")

    # Log the model
    mlflow.sklearn.log_model(svc_model, "SupportVectorClassifier_model")

    # Print the results
    print("\nSupportVector Classifier:")
    print(f"Accuracy: {svc_accuracy:.2f}")
    print(f"Precision: {svc_precision:.2f}")
    print(f"Recall: {svc_recall:.2f}")
    print(classification_report(y_test, svc_predictions))
