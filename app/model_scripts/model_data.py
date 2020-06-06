import re
import pandas as pd
from sqlalchemy import create_engine

import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['wordnet','punkt'])

def tokenize(text):
    text = re.sub(pattern=r'[^A-Za-z0-9]+',repl=' ', string=text.lower().strip())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        new_token = lemmatizer.lemmatize(token)
        clean_tokens.append(new_token)
    
    return clean_tokens

def model_evaluate(query):

    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('Messages', engine)

    model = joblib.load("../models/model.pkl")
    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return classification_labels, classification_results