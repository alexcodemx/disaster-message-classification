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
nltk.download(['wordnet','punkt'])

#python train_classifier.py ../data/DisasterResponse.db model.pkl

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table('Messages',con=engine)
        
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    text = re.sub(pattern=r'[^A-Za-z0-9]+',repl=' ', string=text.lower().strip())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        new_token = lemmatizer.lemmatize(token)
        clean_tokens.append(new_token)
    
    return clean_tokens


def build_model():
    
    pipeline = Pipeline(
    [
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('kn',MultiOutputClassifier(KNeighborsClassifier(n_neighbors=9)))
    ]
    )
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    accuracy = (Y_test==Y_pred).mean()
    print("Accuracy:\n{}".format(accuracy))
    
    for i in range(Y_test.shape[1]):
        print("\nClassification & Accurancy Report for {}".format(Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:,i],pd.DataFrame(Y_pred).iloc[:,i]))


def save_model(model, model_filepath):
    
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