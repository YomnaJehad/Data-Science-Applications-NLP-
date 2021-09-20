import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk
import pickle
import sys
nltk.download('stopwords')
#-----------------------
arabi_text = sys.argv[1]
#------------------------

df = pd.read_csv("oversampling_dataset.csv")
df.head()
#------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['text_normal_tokens'])
#-------------------------


def filtering_sentence(text):
    stop_words= stopwords.words('arabic')
    filtered_sentence = []
    w = text.translate(str.maketrans("","",string.punctuation))
    splitted_sentence=[]
    splitted_sentence = w.split()
    for word in splitted_sentence:
            if word not in stop_words:
                 filtered_sentence.append(word)
    final_string=str(filtered_sentence)
    return final_string


#--------------------------
x=filtering_sentence(arabi_text)
#[df.text_normal_tokens[2]]
text_tf= tf.transform([x])
text_tf

#------------------------------

filename = r'finalized_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict (text_tf)
print(pred)