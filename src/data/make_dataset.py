import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:\\Users\\Abhishek\\Downloads\\Restaurant_Reviews.tsv",sep='\t')

train_data,test_data=train_test_split(df,test_size=0.2,random_state=101)
data_path=os.path.join("data","raw")
os.makedirs(data_path,exist_ok=True)
train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)

