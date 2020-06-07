# importing required dependencies
import pandas as pd
import numpy as np

import re, unicodedata

import nltk
from nltk.stem import wordnet 
from nltk import pos_tag 
from nltk import word_tokenize 
from nltk.corpus import stopwords 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import pairwise_distances 

from python_scripts.helper_functions import *


# Reading data
data_path = '/kaggle/input/cloud-counselage-qa-data/'

data = pd.read_excel(data_path + 'faq_data.xlsx')
data.ffill(axis = 0,inplace=True)

# Data Preparation
data['cleaned_data'] = chunk_clean(data.question.values)
data['norm_data'] = chunk_text_normalize(data.cleaned_data.values)

# Creating word vectorizer object
word_vectorizer = TfidfVectorizer()

# Fitting data into the TfidfVectorizer
faq_data = word_vectorizer.fit_transform(data.norm_data.values).toarray() 

faq_data_features=pd.DataFrame(faq_data,columns=word_vectorizer.get_feature_names()) 
vocab_text = list(faq_data_features.columns)


# Other functions for chatbot
def predict_answer(text):
    text = text_normalization(clean_text(text))
    text = word_vectorizer.transform([text]).toarray()

    cosine_similarity=1-pairwise_distances(faq_data_features,text,metric='cosine')
    output = data['response_text'].loc[cosine_similarity.argmax()]
    
    return output


def chat():
    print('NATUKAKA: Welcome User, I am a chatbot assistant\n')
    while True:
        text = str(input('YOU: '))
        
        if text=='quit':
            print('NATUKAKA: ','Bye, See you again soon','\n')
            break
            
        response = predict_answer(text)
        print('NATUKAKA: ',response,'\n')


chat()