#dep
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
#TF-IDF score to decide the most important word/term in the document (vectorizing the text)
from sklearn.feature_extraction.text import TfidfVectorizer
#to distinguish between true and fake news, we use a magical line inside this class (linear support vector classification)
from sklearn.svm import LinearSVC

#reading file 
data = pd.read_csv("fake_or_real_news.csv")
#creates a column in the dataset based on the label column, 0 if its real and 1 if its fake 
data['fake'] = data['label'].apply(lambda x:0 if x=="REAL" else 1)
data = data.drop("label", axis=1)
# we only need the text not the title 
x, y =data['text'], data['fake']

#to split the dataset into (%80 training, %20 testing):
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
#used for training the vectorizer on the training data and then transforming the training data into its numerical representation
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

#classifier 
clf = LinearSVC()
clf.fit(x_train_vectorized, y_train)
#accuracy percentage :)
clf.score(x_test_vectorized, y_test)


with open("mytext.txt", "w", encoding="utf-8") as f:
    f.write(x_test.iloc[10])
    
with open("mytext.txt", "r", encoding="utf-8") as f:
    text = f.read()
vectorized_text = vectorizer.transform([text])
clf.predict(vectorized_text)

y_test.iloc[10]



