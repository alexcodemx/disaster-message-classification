import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from the csv files.

    Loads data from messages and categories csv files.

    Args:
        messages_filepath: Path to the messages csv file
        categories_filepath: Path to the category csv file

    Returns:
        merged_df: Files merged in a data frame

    """  
    messages = pd.read_csv(messages_filepath, dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    
    merged_df = pd.merge(messages, categories, on="id")
    
    return merged_df


def clean_data(df):
    """Clean data.

    Transforms and cleans the dataframe.

    Args:
        df: Dataframe

    Returns:
        df: Clean dataframe.

    """    
    categories = df.categories.str.split(pat=";", expand=True)
    
    exp = lambda x: x.split("-")[0]
    category_colnames = categories.iloc[1,:].apply(exp).tolist()
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.extract(r'(\d)$')

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop(columns="categories")
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    df = df.drop_duplicates()
        
    return df


def save_data(df, database_filename):
    """Save data.

    Saves the dataframe to a sqlite database.

    Args:
        df: Dataframe
        database_filename: Name for the database

    Returns:
        None

    """  
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()