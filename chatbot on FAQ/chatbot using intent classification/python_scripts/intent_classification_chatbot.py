# importing required dependencies
import os
import json

import tensorflow as tf

from python_scripts.helper_functions import *

# Reading data
data_path = '/kaggle/input/cloud-counselage-qa-data/'
data_json = 'intents2.json'

with open(data_path+data_json) as json_data:
    intents_dict = json.load(json_data)


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


# loading the saved model
model_path = '/kaggle/input/chatbot-with-intent-classification/chatbot8/'
model = tf.keras.models.load_model(model_path)


# Defining functions to predict
def bow(sentence, vector, show_details=False):
    
    sentence_words = text_normalization(clean_text(sentence))
    
    bag = [0]*len(vector)  
    for s in sentence_words.split():
        for i,w in enumerate(vector):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


ERROR_THRESHOLD = 0.25
def classify(sentence):
    sent = pd.DataFrame([bow(sentence,vector=words)],dtype=float,index=['input'])

    results = model.predict([sent])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append((tags[r[0]], r[1]))
    
    return return_list


def predict_response(sentence, show_details=False):
    results = classify(sentence)
    if show_details:
            print('Tag: ',results[0][0])
    if results:
        while results:
            for i in intents_dict['intents']:
                if i['tag'] == results[0][0]:
                    response = random.choice(i['responses'])
                    return response
                

# For interaction with chatbot
def chat():
    print('NATUKAKA: Welcome User, I am a chatbot assistant\n')
    while True:
        text = str(input('YOU: '))
        
        if text=='quit':
            print('NATUKAKA: ','Bye, See you again soon!','\n')
            break
            
        response = predict_response(text)
        print('NATUKAKA: ',response,'\n')
 

chat()

