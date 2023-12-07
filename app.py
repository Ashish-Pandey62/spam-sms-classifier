import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')


count_vectorizer = pickle.load(open('vectorized.pkl','rb'))
model = pickle.load(open('sms_classifier.pkl','rb'))

# we can make our webapp working in these steps :

# 1. text preprocessing : 

import re
from nltk.corpus import stopwords
import string

def pre_process(text):
  final = []

  tokenized = nltk.word_tokenize(text)
  no_special = re.sub('[^A-Za-z0-9.]+', ' ',text)



  no_special = no_special.split()
  stops = stopwords.words('english')
  for word in no_special:
     if word not in stops and word not in string.punctuation:
      final.append(word.lower())

  # for word in final :
  #   final.append(word.lower())



  return final



st.title("SMS_SPAM_PREDICTOR")
user_input = st.text_area('Enter the sms/text/email')

if st.button('Predict'):
    transformed_text = pre_process(user_input)
    # vectorized_text = bnb.transform([transformed_text])
    vectorized_text = count_vectorizer.transform([' '.join(transformed_text)])
    result = model.predict(vectorized_text)[0]
    
    if result == 1:
            st.header("Spam")
    else:
            st.header("Not Spam")

    
    