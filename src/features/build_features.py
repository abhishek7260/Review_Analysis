import numpy as np
import pandas as pd
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load datasets
train_data = pd.read_csv("E:\\review_classification\\data\\raw\\train.csv")
test_data = pd.read_csv("E:\\review_classification\\data\\raw\\test.csv")

# Column containing the text data
TEXT_COLUMN = 'Review'

# Function to remove special characters
def remove_special_char(df):
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    return df

# Function to remove stopwords
def remove_stopwords(df):
    custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                        'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    stop_words = set(stopwords.words("english")) - custom_stopwords
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
        lambda x: ' '.join(word for word in x.split() if word.lower() not in stop_words)
    )
    return df

# Function to perform stemming
def stemming(df):
    ps = PorterStemmer()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
        lambda x: ' '.join(ps.stem(word) for word in x.split())
    )
    return df

# Function to preprocess the text data
def preprocess_data(df):
    df = remove_special_char(df)
    df = remove_stopwords(df)
    df = stemming(df)
    return df

# Apply preprocessing to train and test datasets
processed_train = preprocess_data(train_data)
processed_test = preprocess_data(test_data)

data_path=os.path.join("data","processed")
os.makedirs(data_path,exist_ok=True)
processed_train.to_csv(os.path.join(data_path,"processed_train.csv"),index=False)
processed_test.to_csv(os.path.join(data_path,"processed_test.csv"),index=False)
