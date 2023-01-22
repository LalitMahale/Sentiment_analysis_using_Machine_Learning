import streamlit as st
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
import pickle


model = pickle.load(open(r"C:\Users\lalit\Desktop\Projects\Sentiment Analysis 21_jan\model.sav","rb"))
0
st.title("Sentiment Anlysis")

text = st.text_area("Enter Your text here..")

heart_diagnosis = ''

if st.button('Predict Sentiment'):
    heart_prediction = model.Predict([])                          
    
    if (heart_prediction[0] == 1):
      heart_diagnosis = 'This is Positive sentiment'
    else:
      heart_diagnosis = 'This is Negetive sentiment'
    
    st.success(heart_diagnosis)