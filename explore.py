## imports

import os
import re
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

import wrangle

import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata
import nltk
from wordcloud import WordCloud
from nltk import ngrams

from pprint import pprint

## functions

def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english') # + ADDITIONAL_STOPWORDS
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words


# Define a function to drop words containing 'http'
def drop_http_words(text):
    words = text.split()
    filtered_words = [word for word in words if 'http' not in word]
    return ' '.join(filtered_words)

    # Apply the function to the 'readme_contents' column
    new_df['readme_contents'] = new_df['readme_contents'].apply(drop_http_words)

    # Now the DataFrame 'new_df' will not contain words with 'http' in the 'readme_contents' column
    print(new_df)