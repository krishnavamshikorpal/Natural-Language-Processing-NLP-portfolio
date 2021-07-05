import pandas as pd
import os
os.chdir(r"C:\Users\DEBJYOTI BANERJEE\Documents\ALL excel\fake-and-real-news-dataset")
d=pd.read_csv("Final combined data.csv")
data=pd.DataFrame(d)
print(data.shape)
data=data.drop(columns=["Unnamed: 0"])
print(data.info())
data=data.sample(frac=1).reset_index(drop=True)
print(data.head(10))
data.to_csv(r"C:\Users\DEBJYOTI BANERJEE\Documents\ALL excel\fake-and-real-news-dataset\Shuffled Real & Fake Data.csv")