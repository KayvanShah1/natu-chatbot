# importing required dependencies
import numpy as np
import pandas as pd
import os,re,random
import unicodedata
import json

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,Model

from python_scripts.helper_functions import *


# Reading data
data_path = '/kaggle/input/cloud-counselage-qa-data/'
data_json = 'intents2.json'

with open(data_path+data_json) as json_data:
    intents_dict = json.load(json_data)

    
# Data Preparation
qa_data = []    
for intent in intents_dict['intents']:
    for pattern in intent['patterns']:
        for response in intent['responses']:
            qa_data.append((pattern,response,intent['tag']))
            
data = pd.DataFrame(qa_data,columns=['question','response','tag'])
data['tag'] = pd.Categorical(data['tag'])
data['labels'] = data.tag.cat.codes

# Bag of words
words = []
tags = []
documents = []
responses = []

for intent in intents_dict['intents']:
    for pattern in intent['patterns']:
        pattern = text_normalization(clean_text(pattern))
        word = pattern.split()
        words.extend(word)
        documents.append((word, intent['tag']))
        
    tags.append(intent['tag'])
        
# words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

tags = sorted(list(tags))


# Data Pipeline
training = []
output = []

output_empty = [0] * len(tags)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    output_row = list(output_empty)
    output_row[tags.index(doc[1])] = 1

    training.append([bag, output_row])

np.random.shuffle(training)
training = np.array(training)

# create train and labels lists
train_x = list(training[:,0])
train_y = list(training[:,1])


# Modelling
def build_model(max_length = 1024, num_classes = 100):
    
    input_word_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
            
    out = Dense(64,activation=tf.nn.swish)(input_word_ids)
    out = Dense(32,activation=tf.nn.swish)(out)
    out = Dense(num_classes, activation='softmax')(out)
    
    model = Model(inputs=input_word_ids, outputs=out)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = build_model(max_length=len(train_x[0]),num_classes=len(tags))


# Training the model
EPOCHS = 75
history = model.fit(np.array(train_x),np.array(train_y),epochs=EPOCHS)


# Saving the model
model.save('/kaggle/working/chatbot8')
