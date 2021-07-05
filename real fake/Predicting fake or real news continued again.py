import pandas as pd
import os, re
os.chdir(r"C:\Users\DEBJYOTI BANERJEE\Documents\ALL excel\fake-and-real-news-dataset")
d=pd.read_csv("Shuffled Real & Fake Data.csv")
data=pd.DataFrame(d)
data=data.drop(columns=["Unnamed: 0"])
print(data.columns)
print(data.head(10))
def lower_case(a):
    return a.lower()
data.title=data.title.apply(lower_case)
print(data.title.head(10))

def extra_left_space(title):
    aa=title.lstrip()
    ab=re.sub("\s{2,}"," ",aa)
    return ab
data.title=data.title.apply(extra_left_space)

def remove_numbers(title):
    ac=re.sub("\d+"," ",title)
    ad=re.sub("\s{2,}"," ",ac)
    return ad
data.title=data.title.apply(remove_numbers)

def spelling_correction(text):
    b=re.sub("i'm","i am",text)
    return b
data.title=data.title.apply(spelling_correction)

def spelling_correction_2(text):
    c=re.sub("there's","there is",text)
    return c
data.title=data.title.apply(spelling_correction_2)

def spelling_correction_3(text):
    d=re.sub("i've","i have",text)
    return d
data.title=data.title.apply(spelling_correction_3)

def spelling_correction_4(text):
    e=re.sub("what's","what is",text)
    return e
data.title=data.title.apply(spelling_correction_4)

def spelling_correction_5(text):
    f=re.sub("can't","can not",text)
    return f
data.title=data.title.apply(spelling_correction_5)

def spelling_correction_6(text):
    g=re.sub("don't","do not",text)
    return g
data.title=data.title.apply(spelling_correction_6)

def spelling_correction_7(text):
    h=re.sub("i'll","i will",text)
    return h
data.title=data.title.apply(spelling_correction_7)

def spelling_correction_8(text):
    aaa=re.sub("we're","we are",text)
    return aaa
data.title=data.title.apply(spelling_correction_8)

def spelling_correction_9(text):
    aab=re.sub("it's","it is",text)
    return aab
data.title=data.title.apply(spelling_correction_9)

def char_conversion(text):
    var=re.sub("yr","year",text)
    return var
data.title=data.title.apply(char_conversion)

def improvement(text):
    var1=re.sub("[^a-z]"," ",text)
    var2=var1.lstrip()
    var3=re.sub("\s{2,}"," ",var2)
    return var3
data.title=data.title.apply(improvement)

import nltk.corpus
from nltk.corpus import stopwords
stop=stopwords.words('english')

data.title=data.title.apply(lambda x: 
    ' '.join([word for word in x.split() if word not in (stop)]))

data["title2"]=data.title.apply(lambda x: nltk.word_tokenize(x))

   
from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text

data["title2"]=data["title2"].apply(lambda x: word_lemmatizer(x))
data["title2"]=data["title2"].apply(lambda x: ' '.join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(data["title2"]).toarray()
y=data["target"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model = RandomForestClassifier(n_estimators = 600, 
                            criterion = 'entropy')
                              
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
 
data.to_csv(r"C:\Users\DEBJYOTI BANERJEE\Documents\ALL excel\fake-and-real-news-dataset\Final Analysable data.csv")