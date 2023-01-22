import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from spacy.displacy import render

stop_word = list(STOP_WORDS)
print(stop_word[:10])

# load English pre train small model
nlp = spacy.load("en_core_web_sm")

df1 = pd.read_csv("sentiment labelled sentences/amazon_cells_labelled.txt",sep="\t", header=None)
df2 = pd.read_csv("sentiment labelled sentences/imdb_labelled.txt",sep="\t",header=None)
df3 = pd.read_csv("sentiment labelled sentences/yelp_labelled.txt", sep= "\t",header= None)

df = pd.concat([df1,df2,df3],axis=0)

# Rename the column name
df.rename({0:"Review",1:"Sentiment"},axis=1,inplace=True)


# Now Clean the data

def clean_data(text):
    doc = nlp(text)
    list_tokens = []
    clean_tokens = []
    
    for token in doc:
        if token.lemma_ != "-PRON-":
            tem_token = token.lemma_.lower().strip()
        else:
            tem_token = token.lower()
        list_tokens.append(tem_token)
        
    for token in list_tokens:
        if token not in stop_word and token not in punctuation:
            clean_tokens.append(token)
    
    return clean_tokens


# now we are doing vectorization

tfidf = TfidfVectorizer(tokenizer= clean_data)

l_svm = LinearSVC()

# Split data into x and y 
x = df["Review"]
y = df["Sentiment"]

# Now split data into train test split 

X_train,X_test, y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=2023)

pipe_line = Pipeline([("tfidf",tfidf),("classifier",l_svm)])
pipe_line.fit(X_train,y_train)
Pipeline(memory= None,
        steps = [("tfidf",TfidfVectorizer(analyzer="word", binary=False, decode_error="strict",
                                          encoding = "utf-8", input = "content",
                                         lowercase = True, max_df = 1.0, max_features = None, min_df = 1, ngram_range = (1,1),
                                         norm = "l2", preprocessor = None, smooth_idf = True, stop_words = None, strip_accents = None,
                                         sublinear_tf = False, token_pattern = "(?u)\\u\\w\\w+\\b", 
                                         use_idf = True, vocabulary = None)),
                ("clf",LinearSVC(C = 1.0, class_weight=None, dual= True, fit_intercept=True, intercept_scaling=1,
                                loss = "squared_hinge", max_iter=1000, multi_class="ovr", penalty = 'l2',random_state=None,
                                tol=0.0001, verbose=0))],verbose=False)
y_pred = pipe_line.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(classification_report(y_pred,y_test))

print(pipe_line.predict(["wow, This is amazing not lesson"])[0])

###################
import pickle
model = open("model.sav","wb")
pickle.dump(pipe_line,model)
model = pickle.load(open("model.sav","rb"))
print(model.predict(["my name is khan"]))
