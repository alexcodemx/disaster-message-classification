import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['wordnet','punkt','stopwords'])

def load_data(database_filepath):
    """Load data from the db

    Loads data from the database in a data frame.

    Args:
        database_filepath: Path to the database

    Returns:
        X: Features
        Y: Target variables
        category_names: Target variables column names

    """   
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table('Messages',con=engine)
        
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names


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


def build_model():
    """Build Model

    Defines the model pipeline and the parameters for the grid search.

    Args:
        None

    Returns:
        cv: Grid Search object

    """

    # Pipeline definition. KNeighborsClassifier parameters are predifined based on previous Grid Search runs.
    pipeline = Pipeline(
    [
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('kn',MultiOutputClassifier(KNeighborsClassifier(metric='minkowski', weights='uniform')))
    ]
    )

    # Some parameters were tested before and defined in the pipeline. For speeding up the execution, 
    # few parameters are left for testing.
    parameters = {
        'kn__estimator__n_neighbors':range(5,11,2)
    #    'kn__estimator__weights':['uniform','distance']
    #    'kn__estimator__metric':['minkowski','euclidean']
    }

    # Define the Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate Model

    Tests the the model and prints some metrics.

    Args:
        model: Model object.
        X_test: Testing features.
        Y_test: Testing target variables.
        category_names: Testing target variables column names.

    Returns:
        None

    """

    Y_pred = model.predict(X_test)
    
    accuracy = (Y_test==Y_pred).mean()
    print("Accuracy:\n{}".format(accuracy))
    
    for i in range(Y_test.shape[1]):
        print("\nClassification & Accurancy Report for {}".format(Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:,i],pd.DataFrame(Y_pred).iloc[:,i]))


def save_model(model, model_filepath):
    """Save Model

    Save the model to the specified filepath.

    Args:
        model: Model object.
        model_filepath: Path where the model will be saved.

    Returns:
        None

    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Selecting best estimator...')
        model = model.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()