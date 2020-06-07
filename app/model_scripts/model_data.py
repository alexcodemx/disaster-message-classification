import re
import pandas as pd
from sqlalchemy import create_engine

import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['wordnet','punkt','stopwords'])

def tokenize(text):
    """Tokenize text

    Transform a text into tokens.

    Args:
        text: Text that will be transformed.

    Returns:
        clean_tokens: Tokens from the input text.

    """

    # Replace URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove non alphanumeric characters
    text = re.sub(pattern=r'[^A-Za-z0-9]+',repl=' ', string=text.lower().strip())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in filtered_tokens:
        new_token = lemmatizer.lemmatize(token)
        clean_tokens.append(new_token)
    
    return clean_tokens

def model_evaluate(query):
    """Evaluate the query

    Evaluate the query and deliver the predicted result.

    Args:
        query: Text that will be evaluated.

    Returns:
        classification_labels: Classification labels.
        classification_results: Classification results.

    """
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('Messages', engine)

    model = joblib.load("models/model.pkl")
    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return classification_labels, classification_results